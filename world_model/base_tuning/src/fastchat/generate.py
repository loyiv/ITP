import sys
import json
import os
import fire
import numpy as np
import random
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import (
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    GenerationConfig,
)
from conversation import get_conv_template
from model_adapter import get_model_adapter
from peft_adapter import PeftModelAdapter

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def load_model(
    model_path: str,
    zero_tune: bool = False,
    device: str = "cuda",
    bf16: bool = False,
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    load_8bit: bool = False,
    q_lora: bool = True,
    revision: str = "main",
):

    device_map = {"": device}
    kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "load_in_8bit": load_8bit,
        "revision": revision,
    }

    if device == "cpu":
        kwargs["torch_dtype"] =  torch.float32
    elif device == "cuda":
        kwargs["device_map"] = "auto"
        if num_gpus != 1:
            if max_gpu_memory is None:
                kwargs["device_map"] = "sequential"
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

    if load_8bit and num_gpus != 1:
        raise ValueError("8-bit quantization is not supported for multi-gpu inference.")

    if q_lora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=kwargs["torch_dtype"] ,
        )
    else:
        kwargs["quantization_config"] = None

    if zero_tune:
        model_adapter = get_model_adapter(model_path)
    else:
        model_adapter = PeftModelAdapter()
        if not model_adapter.match(model_path):
            raise ValueError(f"Model path {model_path} does not match the adapter {model_adapter.name}")

    model, tokenizer = model_adapter.load_model(model_path, kwargs)

    model.eval()

    return model, tokenizer

def get_response(output: str, eos_token: str) -> str:
    response = output.split(eos_token)[0].strip()
    return response

def main(
    model_path: str,
    test_data_path: str,
    test_unseen_data_path: Optional[str] = None,
    cache_path: str = "caches",
    output_dir: str = "results",

    zero_tune: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    max_gpu_memory: str = None,
    bf16: bool = False,
    load_8bit: bool = False,
    q_lora: bool = True,

    padding_side: str = "left",
    truncation_side: str = "left",
    cutoff_len: int = 1024,
    template_name: str = "vicuna_v1.1",
    num_proc: int = 8,

    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = -1,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    random_seed: int = 42,
):
    print(
        f"Model generation with params:\n"
        f"model_path: {model_path}\n"
        f"test_data_path: {test_data_path}\n"
        f"test_unseen_data_path: {test_unseen_data_path}\n"
        f"cache_path: {cache_path}\n"
        f"output_dir: {output_dir}\n"
        f"zero_tune: {zero_tune}\n"
        f"batch_size: {batch_size}\n"
        f"device: {device}\n"
        f"max_gpu_memory: {max_gpu_memory}\n"
        f"bf16: {bf16}\n"
        f"load_8bit: {load_8bit}\n"
        f"q_lora: {q_lora}\n"
        f"padding_side: {padding_side}\n"
        f"truncation_side: {truncation_side}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"template_name: {template_name}\n"
        f"num_proc: {num_proc}\n"
        f"max_new_tokens: {max_new_tokens}\n"
        f"do_sample: {do_sample}\n"
        f"temperature: {temperature}\n"
        f"top_p: {top_p}\n"
        f"top_k: {top_k}\n"
        f"num_beams: {num_beams}\n"
        f"repetition_penalty: {repetition_penalty}\n"
        f"random_seed: {random_seed}\n"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if batch_size > 1 and padding_side != "left":
        raise ValueError("Batched inference only supports left padding.")

    set_seed(random_seed)

    model, tokenizer = load_model(
        model_path,
        zero_tune=zero_tune,
        device=device,
        bf16=bf16,
        max_gpu_memory=max_gpu_memory,
        load_8bit=load_8bit,
        q_lora=q_lora,
    )

    if model.config.model_type == "llama" or model.config.model_type == "mistral":
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    print("Tokenizer special tokens: ", tokenizer.special_tokens_map)
    print("Tokenizer padding_side: ", tokenizer.padding_side)

    def tokenize_prompt(data_point):
        conv = get_conv_template(template_name)

        if "instruction" in data_point and str(data_point["instruction"]).strip() != "":
            conv.set_system_message(data_point["instruction"])

        conversations = data_point["conversations"].copy()
        assert conversations[-1]["from"] == "gpt"
        conversations[-1]["value"] = ""

        for utt in conversations:
            if utt["from"] == "human":
                conv.append_message(conv.roles[0], utt["value"])
            else:
                conv.append_message(conv.roles[1], utt["value"])

        prompt = conv.get_prompt()

        tokenized_inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        return tokenized_inputs

    cache_dir = os.path.join(cache_path, "caches_base")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_files = {"test": test_data_path}
    if test_unseen_data_path is not None:
        if os.path.exists(test_unseen_data_path):
            data_files["test_unseen"] = test_unseen_data_path
        else:
            print(f"Ignore: test_unseen_data_path {test_unseen_data_path} does not exist.")

    load_data = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    test_dataset = load_data["test"].map(tokenize_prompt, num_proc=num_proc)

    signature_columns = ["input_ids", "attention_mask"]
    ignored_columns = list(set(test_dataset.column_names) - set(signature_columns))
    test_dataset = test_dataset.remove_columns(ignored_columns)

    all_dataloaders = []

    data_collator = DataCollatorWithPadding(
        tokenizer, padding=True, return_tensors="pt"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2,
    )
    all_dataloaders.append(("test", test_dataloader))

    if "test_unseen" in load_data:
        test_unseen_dataset = load_data["test_unseen"].map(tokenize_prompt, num_proc=num_proc)
        test_unseen_dataset = test_unseen_dataset.remove_columns(ignored_columns)
        test_unseen_dataloader = DataLoader(
            test_unseen_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=2,
        )
        all_dataloaders.append(("test_unseen", test_unseen_dataloader))

    generation_config = GenerationConfig(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )

    for dataloader_tuple in all_dataloaders:
        part, dataloader = dataloader_tuple[0], dataloader_tuple[1]
        if part == "test":
            print("Generating on the test set ...")
            save_gold = load_data["test"]
            save_path = os.path.join(output_dir, test_data_path.split('/')[-1].split('.')[0] + '_output.jsonl')
        else:
            print("Generating on the test-unseen set ...")
            save_gold = load_data["test_unseen"]
            save_path = os.path.join(output_dir, test_unseen_data_path.split('/')[-1].split('.')[0] + '_output.jsonl')

        results = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch_input_ids = batch["input_ids"].to(device)
                batch_attention_mask = batch["attention_mask"].to(device)
                input_max_len = batch_input_ids.size(1)

                generation_output = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    generation_config=generation_config
                )
                batch_output_ids = generation_output.sequences

                batch_generated_ids = batch_output_ids[:, input_max_len:]

                batch_output = [tokenizer.decode(output_ids) for output_ids in batch_generated_ids]
                batch_result = [get_response(output, eos_token=tokenizer.eos_token) for output in batch_output]
                results.extend(batch_result)

        with open(save_path, 'w', encoding='utf-8') as f:
            for item, res in zip(save_gold, results):
                result = {
                    "dialog_id": item["dialog_id"],
                    "turn_id": item["turn_id"],
                    "gold_response": item["conversations"][-1]["value"],
                    "generated_response": res,
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print("Saved {} resulted examples to {}.".format(len(results), save_path))

if __name__ == "__main__":
    fire.Fire(main)
