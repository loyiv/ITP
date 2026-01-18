
model_name="Mistral-7B-v0.1"

CUDA_VISIBLE_DEVICES=6 python src/fastchat/generate.py \
    --model_path logs/base_${model_name} \
    --test_data_path data/test.json \
    --test_unseen_data_path data/test_unseen.json \
    --output_dir results/base_${model_name} \
    --bf16 False \
    --batch_size 8 \
    --cutoff_len 1024 \
    --max_new_tokens 100 \
    --temperature 0.5 \
    --top_p 0.75 \
    --top_k 40 \
    --random_seed 42
