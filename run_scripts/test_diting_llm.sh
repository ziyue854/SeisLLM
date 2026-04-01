NUM_GPUS=1
MODEL_NAME="SeisLLM_emg"
CHECKPOINT_DIR="/path/to/your/checkpoint" 

TMPDIR=/data/xzy/tmp_cache CUDA_VISIBLE_DEVICES=6 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=29506 \
    main.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    \
    --seed 3407 \
    --mode "test" \
    --model-name $MODEL_NAME \
    --checkpoint  $CHECKPOINT_DIR \
    --use-torch-compile False \
    --use-flash-attn True \
    --llm "GPT2" \
    --LLM_layer_num 6 \
    --d_model 768 \
    --d_patch 256 \
    \
    --log-base "./logs" \
    --log-step 300 \
    \
    --data "./datasets/DiTing330km" \
    --dataset-name "diting_light" \
    --data-split True \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle True \
    --workers 6 \
    --in-samples 8192 \
    \
    --batch-size 192 \
    > ./logs/test_SeisLLM_emg.log 2>&1 &