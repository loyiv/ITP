#!/usr/bin/env bash
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/base_model}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/worldmodel_train.jsonl}"

RUN_NAME="${RUN_NAME:-wm_run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"

DS_CONFIG="${DS_CONFIG:-${ROOT_DIR}/config/deepspeed_config_s2.json}"
MASTER_PORT="${MASTER_PORT:-29400}"

CUTOFF_LEN="${CUTOFF_LEN:-2048}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WD="${WD:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
GRAD_CKPT="${GRAD_CKPT:-False}"

FP16="${FP16:-True}"
BF16="${BF16:-False}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MERGE_LORA="${MERGE_LORA:-true}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_full}"

Q_LORA="${Q_LORA:-false}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

mkdir -p "${OUTPUT_DIR}"

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[FAIL] deepspeed command not found. 请先安装/激活包含 deepspeed 的环境。" >&2
  echo "       例如：pip install deepspeed  或 conda activate <env_with_deepspeed>" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[FAIL] MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[FAIL] DATA_PATH not found: ${DATA_PATH}" >&2
  echo "       需要提供 worldmodel 训练数据（JSON/JSONL），且每条包含 state/action/next_state。" >&2
  exit 1
fi
if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "[FAIL] DS_CONFIG not found: ${DS_CONFIG}" >&2
  exit 1
fi

python - <<PY
import os, json
import transformers

mp = os.environ.get("MODEL_PATH", "${MODEL_PATH}")
dp = os.environ.get("DATA_PATH", "${DATA_PATH}")
print("[check] transformers:", transformers.__version__)
try:
    import deepspeed
except Exception as e:
    raise SystemExit(f"[FAIL] python cannot import deepspeed: {e}")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, legacy=False)
    print("[check] tokenizer loaded. eos_token:", tok.eos_token, "pad_token:", tok.pad_token)
except Exception as e:
    raise SystemExit(f"[FAIL] cannot load tokenizer from {mp}: {e}")

def load_one(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096).strip()
    if not head:
        raise ValueError("empty data file")
    if head[0] == "[":
        data = json.load(open(path, "r", encoding="utf-8"))
        return data[0]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError("no valid jsonl line")

ex = load_one(dp)
need = {"state","action","next_state"}
missing = need.difference(set(ex.keys()))
if missing:
    raise SystemExit(f"[FAIL] data sample missing keys: {missing}. got keys={list(ex.keys())[:20]}")
print("[OK] data schema looks like worldmodel:", sorted(need))
PY

echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] DATA_PATH=${DATA_PATH}"
echo "[info] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[info] DS_CONFIG=${DS_CONFIG}"

deepspeed --master_port="${MASTER_PORT}" src/fastchat/finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --cutoff_len "${CUTOFF_LEN}" \
  --template_name "vicuna_v1.1" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --gradient_checkpointing "${GRAD_CKPT}" \
  --num_train_epochs "${EPOCHS}" \
  --evaluation_strategy "no" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --fp16 "${FP16}" \
  --bf16 "${BF16}" \
  --lr_scheduler_type "cosine" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --q_lora "${Q_LORA}" \
  --merge_lora "${MERGE_LORA}" \
  --merged_lora_dir "${MERGED_DIR}" \
  --deepspeed "${DS_CONFIG}"

echo "[OK] training finished. OUTPUT_DIR=${OUTPUT_DIR}"
echo "[OK] merged model (if enabled) at: ${MERGED_DIR}"


