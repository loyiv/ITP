# -*- coding: utf-8 -*-
import os
import json
import argparse

from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

SYSTEM_PROMPT = "You are a world model. Predict the NEXT STATE textually."
ASSISTANT_HEAD = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def build_prompt(state_text: str, action_text: str) -> str:
    user = (
        "STATE:\n" + state_text + "\n\n"
        + "ACTION:\n" + action_text + "\n\n"
        + "Please write the NEXT STATE (observation, inventory, brief outcome)."
    )

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        + user
        + ASSISTANT_HEAD
    )

class WMJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = 0):
        self.rows: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                prompt = build_prompt(d["state"], d["action"])
                target = d["next_state"]
                self.rows.append(
                    {
                        "prompt": prompt,
                        "target": target,
                        "meta": {
                            "traj_type": d.get("traj_type", "expert"),
                        },
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

class PromptOnlyLossCollator:
    def __init__(self, tokenizer, max_len: int = 2048):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]):
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        tok_prompt = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
        )
        tok_target = self.tok(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
        )

        input_ids_list = []
        labels_list = []
        attn_list = []

        for i in range(len(prompts)):
            p_ids = tok_prompt["input_ids"][i]
            t_ids = tok_target["input_ids"][i]

            ids = torch.cat([p_ids, t_ids], dim=0)

            lab = torch.cat(
                [torch.full_like(p_ids, -100), t_ids.clone()],
                dim=0,
            )
            att = torch.ones_like(ids)

            if ids.size(0) > self.max_len:
                ids = ids[-self.max_len:]
                lab = lab[-self.max_len:]
                att = att[-self.max_len:]

            input_ids_list.append(ids)
            labels_list.append(lab)
            attn_list.append(att)

        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        maxL = max(x.size(0) for x in input_ids_list)

        def pad_stack(tensors, pad_val):
            out = torch.full(
                (len(tensors), maxL), pad_val, dtype=tensors[0].dtype
            )
            for i, t in enumerate(tensors):
                out[i, -t.size(0):] = t
            return out

        input_ids = pad_stack(input_ids_list, pad_id)
        labels = pad_stack(labels_list, -100)
        attention_mask = pad_stack(attn_list, 0)

        batch_out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        return batch_out

def build_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def _to_torch_dtype(name: Optional[str]) -> torch.dtype:
    if not name:
        return torch.float16
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float16

def build_model(model_path: str, quant_cfg: Optional[Dict] = None):
    quant_cfg = quant_cfg or {}
    dtype = _to_torch_dtype(quant_cfg.get("torch_dtype", "float16"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )

    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    model.to(device)

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def default_lora_cfg():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/world_model.yaml",
        help="world model 训练配置文件",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_jsonl = cfg["train_jsonl"]
    model_path = cfg["model_path"]
    output_dir = cfg["output_dir"]
    merged_dir = cfg["merged_dir"]
    quant_cfg = cfg.get("quantization", {}) or {}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    max_samples = int(cfg.get("max_samples", 0) or 0)
    dataset = WMJsonlDataset(train_jsonl, max_samples=max_samples)
    print(f"[train_wm] loaded {len(dataset)} samples from {train_jsonl}")

    tokenizer = build_tokenizer(model_path)
    model = build_model(model_path, quant_cfg)
    lora_cfg = default_lora_cfg()
    collator = PromptOnlyLossCollator(
        tokenizer,
        max_len=int(cfg.get("max_seq_len", 2048)),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.get("epochs", 3)),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type="constant",
        logging_steps=int(cfg.get("logging_steps", 25)),
        save_steps=int(cfg.get("save_steps", 500)),
        optim="adamw_torch",
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_cfg,
        max_seq_length=int(cfg.get("max_seq_len", 2048)),
        data_collator=collator,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_wm] LoRA checkpoint saved at {output_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if hasattr(base_model.config, "attn_implementation"):
        base_model.config.attn_implementation = "eager"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    merged_model = PeftModel.from_pretrained(base_model, output_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[train_wm] merged full model saved at {merged_dir}")

if __name__ == "__main__":
    main()
