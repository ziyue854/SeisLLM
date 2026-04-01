import os
import torch
from config import Config
from models import create_model,load_checkpoint
from utils import *
from .preprocess import SeismicDataset
from .validate import validate
import deepspeed


def test_worker(args,device):
    # Log
    logger.set_logger("test")

    # Data loader
    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)
    test_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="test",
    )

    test_sampler = (
        torch.utils.data.DistributedSampler(test_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=test_sampler,
    )

    logger.info("Running test in DeepSpeed mode...")

    true_sampling_rate = 50 
    if hasattr(test_dataset, 'sampling_rate') and callable(test_dataset.sampling_rate):
        true_sampling_rate = test_dataset.sampling_rate()
        
    # Model
    model = create_model(
            model_name=args.model_name,
            in_channels=in_channels,
            in_samples=args.in_samples,
            llm_type=args.llm,
            llm_layers=args.LLM_layer_num,
            d_model=args.d_model,
            d_patch=args.d_patch,
            d_ff=args.d_ff,
            dataset_name = args.dataset_name,
            n_heads=args.n_heads,
            sampling_rate=true_sampling_rate,
            use_flash_attn_2=args.use_flash_attn,
        )

    model_engine, _, _, _ = deepspeed.initialize(
        args=args, model=model
    )
    device = model_engine.device
    model = model_engine

    if not args.checkpoint:
        raise ValueError("Checkpoint must be provided for testing in DeepSpeed mode.")
    load_path, client_state = model.load_checkpoint(args.checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False)
    if is_main_process():
        epoch_loaded = client_state.get('epoch', 'N/A')
        logger.info(f"Successfully loaded model from checkpoint: {load_path} (Epoch: {epoch_loaded})")
    # logger.info(f"Model loaded via DeepSpeed from: {args.checkpoint}")

    # Loss
    loss_fn = Config.get_loss(model_name=args.model_name).to(device)

    test_loss, test_metrics_dict = validate(
        args, model_tasks, model, loss_fn, test_loader, 0, device, testing=True
    )

    if is_main_process():
        # Metrics merged
        test_metrics_str = "* "
        for task in model_tasks:
            test_metrics_str += f"[{task.upper()}]{test_metrics_dict[task]} "
        logger.info(test_metrics_str)

    return test_loss
