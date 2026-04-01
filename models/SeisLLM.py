from math import sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo.config
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
import transformers
from functools import partial
from ._factory import register_model
from utils import *
import json
import numpy as np

torch._dynamo.config.cache_size_limit = 1024


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, stride, dropout, in_channels=3):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_size = patch_size
        self.stride = stride
        #self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        #self.value_embedding = TokenEmbedding(patch_size, d_model)
        self.projection = nn.Conv1d(
            in_channels=in_channels * patch_size,
            out_channels=d_model,
            kernel_size=1
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        #x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[3], x.shape[2])
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # # Input encoding
        # x = self.value_embedding(x)
        return self.dropout(x), n_vars


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.
    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.
    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.

    Args:
        x (torch.Tensor): N-dimensional tensor.
        input (Tensor): N-dimensional tensor
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
        dim (int): Dimension to pad.
        padding_value (float): fill value.

    Raises: AssertionError: `kernel_size` is less than `stride`.

    Returns: torch.Tensor : padded tensor.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )
        
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, 
                              stride=stride, bias=False)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Multi_Scale_Conv_Block(nn.Module):
    def __init__(
        self, scale_num, scale_stride, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num)
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=scale_num * out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list()
        for conv in self.convs:
            xi = conv(x)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)
    
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class LLM_Block(nn.Module):
    def __init__(self, llm, layers, use_flash_attn_2):
        super(LLM_Block, self).__init__()

        self.llm_model_type = llm

        self._initialize_llm_model(layers, use_flash_attn_2)
        self._setup_padding_token()
        self._freeze_params()

    def _initialize_llm_model(self, layers, use_flash_attn_2):
        if self.llm_model_type == "LLAMA":
            self._initialize_llama(layers, use_flash_attn_2)
        elif self.llm_model_type == "GPT2":
            self._initialize_gpt2(layers)
        elif self.llm_model_type == "BERT":
            self._initialize_bert(layers)
        else:
            raise Exception('LLM model is not defined')
        
    def _initialize_llama(self, layers, use_flash_attn_2):
        self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.d_llm = self.llama_config.hidden_size

        attn_impl = "flash_attention_2" if use_flash_attn_2 else "sdpa"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.llm_model = LlamaModel.from_pretrained(
            'huggyllama/llama-7b',
            trust_remote_code=True,
            config=self.llama_config,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(
            'huggyllama/llama-7b',
            trust_remote_code=True
        )
        

    def _initialize_gpt2(self, layers):
        self.gpt2_config = GPT2Config.from_pretrained("./pretrained_models/gpt2") #'openai-community/gpt2'
        self.gpt2_config.num_hidden_layers = layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True
        self.d_llm = self.gpt2_config.n_embd

        self.llm_model = GPT2Model.from_pretrained(
            "./pretrained_models/gpt2",
            trust_remote_code=True,
            config=self.gpt2_config,
        )
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "./pretrained_models/gpt2",
            trust_remote_code=True
        )
        

    def _initialize_bert(self, layers):
        self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        self.bert_config.num_hidden_layers = layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True
        self.d_llm = self.bert_config.hidden_size
        
        self.llm_model = BertModel.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True,
            config=self.bert_config,
        )
        
        self.tokenizer = BertTokenizer.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True
        )
        

    def _setup_padding_token(self):
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
    def _freeze_params(self):
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()
    
    def forward(self, inputs_embeds, attention_mask=None):
        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )


class HeadDetectionPicking(nn.Module):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels,
        layer_channels,  # dp_head_channels: [128, 160, 192, 224]
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)

        self.depth = len(layer_channels)

        self.up_layers = nn.ModuleList()

        for inc, outc, kers in zip(
                [feature_channels] + layer_channels[:-1],
                layer_channels[:-1] + [out_channels * 2],
                layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        N, C, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class LayerNormForConv1d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (N, C, L)
        x = x.permute(0, 2, 1)  # -> (N, L, C)
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)  # -> (N, C, L)
        return x

class HeadRegression(nn.Module):
    """Head of regression."""

    def __init__(self, feature_channels, out_act_layer, dropout=0.2, act_layer=nn.GELU, **kwargs):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(feature_channels, feature_channels, 8, 2),
            LayerNormForConv1d(feature_channels),
            act_layer()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(feature_channels, feature_channels, 8, 2),
            LayerNormForConv1d(feature_channels),
            act_layer()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(feature_channels , 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class SeisLLM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=[8, 6, 4, 2, 1],
        conv_channels=[16, 48, 96, 128],
        conv_kernel_sizes=[16, 8, 6, 4, 1],
        conv_strides=[2, 2, 2, 2, 1],
        llm_type="GPT2",
        llm_layers = 6,
        d_model=768,
        d_patch=256,
        d_ff=512,
        dropout=0.2,
        attention_dropout=0.2,
        patch_size=16,
        stride=8,
        n_heads=8,
        dp_head_channels = [128, 160, 192, 224],
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        output_head=HeadRegression,
        dataset_name="diting_light",
        sampling_rate=50,
        task_type=None,
        use_flash_attn_2=False,
        **kwargs
    ):
        super().__init__()

        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels.append(d_model // patch_size)
        # conv_channels.append(d_patch // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.stride = stride
        self.d_patch = d_patch
        self.d_ff = d_ff
        self.top_k = 5
        self.conv_out_channel = d_model // patch_size

        # Multi-Scale Convolutional Embedder
        self.convs = nn.Sequential(
            *[
                Multi_Scale_Conv_Block(
                    scale_num=conv_scale_num,
                    scale_stride=ss,
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for ss, inc, outc, kers, strd in zip(
                    conv_scale_strides,
                    [in_channels] + conv_channels[:-1],
                    conv_channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )

        self.llm_block = LLM_Block(
            llm=llm_type,
            layers=llm_layers,
            use_flash_attn_2=use_flash_attn_2
        )
        self.d_llm = self.llm_block.d_llm
        #self.llm_proj = nn.Linear(self.d_llm, self.d_ff)
        self.feature_channels = self.d_llm

        self.description = load_content(dataset_name)

        self.sampling_rate = sampling_rate
        self.task_type = task_type
        
        self.post_llm_dropout = nn.Dropout(dropout)

        self.patch_embedding = PatchEmbedding(
            self.d_patch, self.patch_size, self.stride, dropout, self.conv_out_channel)
        
        self.word_embeddings = self.llm_block.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(self.d_patch, n_heads, self.d_ff, self.d_llm, attention_dropout)

        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel)
                out_layer_kernel_sizes.insert(0, kernel)

            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias
            )
        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x, metadata):
        x_input = x

        #x = self.normalize_layers(x, 'norm')

        prompt_embeddings, prompt_attention_mask = self._build_prompt_embeddings(x, metadata)

        # Multi Scale Conv Embedder
        convs_out = self.convs(x)
        # convs_out = convs_out.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # convs_out = rearrange(convs_out, 'b c n p -> b n (c p)')

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out, n_vars = self.patch_embedding(convs_out)   # .to(torch.bfloat16)

        patch_nums = enc_out.shape[1]

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        waveform_attention_mask = torch.ones(
            enc_out.shape[0], enc_out.shape[1], 
            device=enc_out.device, 
            dtype=torch.long
        )

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, waveform_attention_mask], dim=1)

        dec_out = self.llm_block(inputs_embeds=llama_enc_out, attention_mask=attention_mask).last_hidden_state
        #dec_out = self.llm_proj(dec_out)
        #dec_out = self.post_llm_dropout(dec_out)
        #dec_out = dec_out[:, :, :self.d_ff]

        dec_out = dec_out[:, -patch_nums:, :]
        dec_out = dec_out.transpose(1, 2)  # [B, d_llm, patch_num]
        # dec_out = rearrange(dec_out, 'b n (c p) -> b c (n p)', p=self.patch_size)
        # Output head
        dec_out = self.out_head(dec_out, x_input)
        return dec_out
    
    def _build_prompt_embeddings(self, x_enc, metadata_batch):
        B, C, L = x_enc.size()
        # x_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=-1)[0]
        max_values = torch.max(x_enc, dim=-1)[0]
        medians = torch.median(x_enc, dim=-1).values

        if self.task_type is not None:
            if self.task_type.lower() == 'dpk':
                task_description = "picking p-wave and p-wave phases based on 3-component seismic waveform data"
            elif self.task_type.lower() == 'pmp':
                task_description = "determining p-wave first-motion polarity from 3-component seismic waveform data"
            elif self.task_type.lower() == 'emg':
                task_description = "estimating earthquake magnitude based on 3-component seismic waveform data"
            elif self.task_type.lower() == 'baz':
                task_description = "estimating earthquake back-azimuth based on 3-component seismic waveform data"
            elif self.task_type.lower() == 'dis':
                task_description = "estimating the epicentral distance based on 3-component seismic waveform data"
        else:
            task_description = "performing seismological analysis based on 3-component seismic waveform data"

        prompt = []
        for b in range(x_enc.shape[0]):
            min_vals_per_channel = [f"{v:.2f}" for v in min_values[b]]
            min_vals_str = f"Z:{min_vals_per_channel[0]}, N:{min_vals_per_channel[1]}, E:{min_vals_per_channel[2]}"
            
            max_vals_per_channel = [f"{v:.2f}" for v in max_values[b]]
            max_vals_str = f"Z:{max_vals_per_channel[0]}, N:{max_vals_per_channel[1]}, E:{max_vals_per_channel[2]}"
            
            median_vals_per_channel = [f"{v:.2f}" for v in medians[b]]
            median_vals_str = f"Z:{median_vals_per_channel[0]}, N:{median_vals_per_channel[1]}, E:{median_vals_per_channel[2]}"
            
            prompt_ = f"""<|start_prompt|>
                        ## Context
                        - Dataset description: {self.description}
                        - sampling rate {self.sampling_rate} Hz

                        ## Task
                        - Task description: {task_description}

                        ## Input statistics
                        - Min Values per channel are [{min_vals_str}]
                        - Max Values per channel are [{max_vals_str}]
                        - Mean Values per channel are [{median_vals_str}]
                        <|<end_prompt>|>"""
            prompt.append(prompt_)

        prompt_tokenized = self.llm_block.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        )
        prompt_ids = prompt_tokenized.input_ids.to(x_enc.device)
        prompt_attention_mask = prompt_tokenized.attention_mask.to(x_enc.device)
        prompt_embeddings = self.llm_block.get_input_embeddings()(prompt_ids)  # (batch, prompt_token, dim)

        #x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        return prompt_embeddings, prompt_attention_mask
    
    def _calculate_lags(self, x_enc):
        n = x_enc.shape[1]

        original_dtype = x_enc.dtype
        x_enc_fp32 = x_enc.float()
        
        x_fft = torch.fft.rfft(x_enc_fp32, n=n, dim=-1)
        acf_fft = x_fft * torch.conj(x_fft)
        corr = torch.fft.irfft(acf_fft, n=n, dim=-1)

        corr = corr.to(original_dtype)
        num_lags_available = corr.size(-1)
        safe_k = min(self.top_k, num_lags_available)
        if safe_k < 1:
            return torch.tensor([], device=x_enc.device, dtype=torch.long).reshape(x_enc.shape[0], 0)
            
        _, lags = torch.topk(corr, safe_k, dim=-1)
        
        return lags


@register_model
def SeisLLM_dpk(**kwargs):
    """Detection and Phase-Picking."""
    model = SeisLLM(
        dropout=0.3,
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        task_type='dpk',
        **kwargs,
    )
    return model


@register_model
def SeisLLM_emg(**kwargs):
    """Magnitude estimation."""
    model = SeisLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
            dropout=0.2,
            act_layer=nn.GELU,
        ),
        task_type='emg',
        dropout=0.2,
        attention_dropout=0.2,
        **kwargs,
    )
    return model

@register_model
def SeisLLM_dis(**kwargs):
    """Epicentral distance estimation."""
    model = SeisLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        task_type='dis',
        **kwargs,
    )
    return model
