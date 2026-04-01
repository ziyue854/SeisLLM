NUM_GPUS=1
MODEL_NAME="SeisLLM_dpk"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=29507 \
    main.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    \
    --seed 3407 \
    --mode "train" \
    --model-name $MODEL_NAME \
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
    --workers 6 \
    --in-samples 8192 \
    \
    --batch-size 160 \
    --augmentation false \
    --epochs 200 \
    --patience 30 \
    \
    > ./train_diting_dpk.log 2>&1 &