

import logging
import pathlib
import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch
try:

    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ModuleNotFoundError:
    zero = None
    ZeroParamStatus = None

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    Trainer,
)
try:

    from transformers.integrations import deepspeed as hf_deepspeed
except Exception:
    hf_deepspeed = None

from data_utils import rank0_print, make_supervised_data_module

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="yahma/llama-7b-hf")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    cache_path: str = field(
        default="caches", metadata={"help": "Path to cache all data."}
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "Padding side (right or left) for padding to max_length"}
    )
    truncation_side: str = field(
        default="left",
        metadata={"help": "Truncation_side (right or left) for input sequences"}
    )
    cutoff_len: int = field(
        default=800,
        metadata={"help": "Sequences will be possibly truncated if longer than cutoff_len."}
    )
    template_name: str = field(
        default="vicuna_v1.1",
        metadata={"help": "Template name for the conversation."}
    )
    mask_dtype: str = field(
        default="bool",
        metadata={"help": "Data type of attention masks."}
    )
    num_proc: int = field(
        default=8, metadata={"help": "Number of processes for data loading."}
    )

@dataclass
class FinetuningArguments(TrainingArguments):

    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model checkpoints will be written."}
    )
    load_in_8bit: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    num_train_epochs: float = 3.0
    evaluation_strategy: Optional[str] = "no"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch"
    fp16: bool = True
    bf16: bool = False
    lr_scheduler_type: str = "linear"
    logging_steps: int = 100
    save_strategy: Optional[str] = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    local_rank: int = field(default=0, metadata={"help": "Local rank of the process."})
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    merge_lora: bool = field(
        default=False,
        metadata={"help": "Merge LoRA adapter back into the base model after training."},
    )
    merged_lora_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory used to store the merged full-precision model. "
            "Defaults to <output_dir>/merged if not set."
        },
    )

def maybe_zero_3(param):

    if zero is None or ZeroParamStatus is None:
        return param.detach().cpu().clone()

    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def train():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, FinetuningArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    try:
        from peft import (
            LoraConfig,
            PeftConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: peft. Please install it (e.g. `pip install peft`)."
        ) from e

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if lora_args.q_lora:
        zero3_enabled = bool(hf_deepspeed is not None and hf_deepspeed.is_deepspeed_zero3_enabled())
        if len(training_args.fsdp) > 0 or zero3_enabled:
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    quantization_config = None
    if lora_args.q_lora:

        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise ImportError(
                "QLoRA requires `transformers` with BitsAndBytesConfig and `bitsandbytes` installed."
            ) from e
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if "chatglm" in model_args.model_name_or_path.lower():
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=compute_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=["query_key_value"],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=compute_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    if not ddp and torch.cuda.device_count() > 1:

        model.is_parallelizable = True
        model.model_parallel = True

    model = get_peft_model(model, lora_config)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=data_args.padding_side,
        truncation_side=data_args.truncation_side,
        trust_remote_code=True,
        use_fast=False,
        legacy=False,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    rank0_print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    rank0_print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    rank0_print(f"Tokenizer unk_token_id: {tokenizer.unk_token_id}")
    rank0_print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")
    rank0_print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if hf_deepspeed is not None and hf_deepspeed.is_deepspeed_zero3_enabled():

        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:

        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        should_merge_lora = (
            lora_args.merge_lora and lora_args.lora_r is not None and lora_args.lora_r > 0
        )
        if should_merge_lora:
            merged_dir = lora_args.merged_lora_dir or os.path.join(
                training_args.output_dir, "merged"
            )
            os.makedirs(merged_dir, exist_ok=True)
            rank0_print(f"Merging LoRA adapter into base model. Saving to {merged_dir}")

            merged_model = None
            merge_source = getattr(trainer, "model", None)
            if merge_source is not None and hasattr(merge_source, "module"):
                merge_source = merge_source.module
            if merge_source is not None and hasattr(merge_source, "merge_and_unload"):
                try:
                    merged_model = merge_source.merge_and_unload()
                except Exception as merge_error:
                    rank0_print(
                        "Direct merge from the training model failed, will reload weights before merging."
                    )
                    merged_model = None

            if merged_model is None:
                peft_config = PeftConfig.from_pretrained(training_args.output_dir)
                target_dtype = compute_dtype
                if (not torch.cuda.is_available()) and target_dtype == torch.float16:
                    target_dtype = torch.float32
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    torch_dtype=target_dtype,
                    trust_remote_code=True,
                )
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    training_args.output_dir,
                )
                merged_model = peft_model.merge_and_unload()

            merged_model.save_pretrained(
                merged_dir,
                safe_serialization=training_args.save_safetensors,
            )
            tokenizer.save_pretrained(merged_dir)
            rank0_print("Merged full weights have been saved.")

if __name__ == "__main__":
    train()
