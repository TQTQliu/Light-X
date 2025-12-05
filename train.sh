export OUTPUT_DIR="output_train"
export LOG_DIR="${OUTPUT_DIR}/stdout_log"
mkdir -p "$LOG_DIR"

huggingface-cli download \
    alibaba-pai/CogVideoX-Fun-V1.1-5b-InP \
    --local-dir ./checkpoints/CogVideoX-Fun-V1.1-5b-InP \
    --local-dir-use-symlinks False
export MODEL_NAME="$(pwd)/checkpoints/CogVideoX-Fun-V1.1-5b-InP"

huggingface-cli download \
    TrajectoryCrafter/TrajectoryCrafter \
    --local-dir ./checkpoints/TrajectoryCrafter \
    --local-dir-use-symlinks False
export TRANSFORMER_PATH="$(pwd)/checkpoints/TrajectoryCrafter"

export DATASET_META_NAME="./data/metadata.json" # modify here
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export DATASET_NAME=""

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=32 \
  --num_train_epochs=10 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --use_deepspeed \
  --train_mode="inpaint" \
  --trainable_modules "." 