
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-qwen3_8B}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/base_model}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${MODEL_NAME}}"
MASTER_PORT="${MASTER_PORT:-29400}"

deepspeed --master_port="${MASTER_PORT}" src/fastchat/finetune.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --cutoff_len 1024 \
    --template_name "vicuna_v1.1" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing False \
    --num_train_epochs 3 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --fp16 True \
    --bf16 False \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora false \
    --merge_lora true \
    --merged_lora_dir "${OUTPUT_DIR}/merged_full" \
    --deepspeed config/deepspeed_config_s2.json
