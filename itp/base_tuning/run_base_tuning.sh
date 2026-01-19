set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-policy_sft_run}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/policy_base_model}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/policy_sft.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${MODEL_NAME}}"
MASTER_PORT="${MASTER_PORT:-29400}"

FASTCHAT_FINETUNE="${REPO_ROOT}/world_model/base_tuning/src/fastchat/finetune.py"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/world_model/base_tuning/config/deepspeed_config_s2.json}"

deepspeed --master_port="${MASTER_PORT}" "${FASTCHAT_FINETUNE}" \
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
    --deepspeed "${DS_CONFIG}"
