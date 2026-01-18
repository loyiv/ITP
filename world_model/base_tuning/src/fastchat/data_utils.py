import json
import os
from typing import Dict, List

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from conversation import SeparatorStyle, get_conv_template

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

WM_SYSTEM_PROMPT = "You are a world model. Predict the NEXT STATE textually."
WM_ASSISTANT_HEAD = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

def _is_world_model_sample(sample: Dict) -> bool:
    if not isinstance(sample, dict):
        return False
    need = {"state", "action", "next_state"}
    return need.issubset(set(sample.keys()))

def _build_world_model_prompt(state_text: str, action_text: str) -> str:
    user = (
        "STATE:\n" + str(state_text) + "\n\n"
        + "ACTION:\n" + str(action_text) + "\n\n"
        + "Please write the NEXT STATE (observation, inventory, brief outcome)."
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + WM_SYSTEM_PROMPT
        + "\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        + user
        + "\n"
        + WM_ASSISTANT_HEAD
    )

def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)

class Preprocessor(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 1024,
                 template_name: str = "vicuna_v1.1",
                 mask_dtype: str = "bool"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_name = template_name
        self.mask_dtype = mask_dtype

    def preprocess_world_model(self, source: Dict) -> Dict:
        prompt = _build_world_model_prompt(source["state"], source["action"])

        next_state = source["next_state"]
        if next_state is None:
            next_state = ""
        next_state = str(next_state)
        if not next_state.rstrip().endswith("<|eot_id|>"):
            target_text = next_state.rstrip() + "\n<|eot_id|>"
        else:
            target_text = next_state

        full_text = prompt + target_text

        tok_full = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = tok_full.input_ids[0]

        if self.mask_dtype == "bool":
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        tok_tgt = self.tokenizer(
            target_text,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tgt_len = len(tok_tgt["input_ids"])

        labels = input_ids.clone()

        labels = torch.where(attention_mask.bool(), labels, torch.full_like(labels, IGNORE_TOKEN_ID))

        real_pos = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
        if real_pos.numel() == 0:
            labels[:] = IGNORE_TOKEN_ID
        else:
            keep = min(int(tgt_len), int(real_pos.numel()))
            if keep <= 0:
                labels[:] = IGNORE_TOKEN_ID
            else:
                cut = int(real_pos.numel()) - keep
                if cut > 0:
                    to_ignore = real_pos[:cut]
                    labels[to_ignore] = IGNORE_TOKEN_ID

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def preprocess_by_round(self, source) -> Dict:
        conv = get_conv_template(self.template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if "instruction" in source and str(source["instruction"]).strip() != "":
            conv.set_system_message(source["instruction"])

        source_convs = source["conversations"]
        if roles[source_convs[0]["from"]] != conv.roles[0]:

            source_convs = source_convs[1:]

        conv.messages = []
        for j, sentence in enumerate(source_convs):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{source}"
            message = sentence["value"]
            if message == '':
                message = '<none>'
            conv.append_message(role, message)

        prompt = conv.get_prompt()

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids[0]
        target = input_ids.clone()

        conversation = self.tokenizer.decode(input_ids)

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
        sep = conv.roles[1] + ": "
        target_turn = conversation.split(sep)[-1].split(conv.sep2)[0] + conv.sep2
        user_prompt = conversation.split(target_turn)[0]
        user_prompt = user_prompt.rstrip()

        instruction_len = len(self.tokenizer(user_prompt).input_ids) - 1
        target_len = len(self.tokenizer(target_turn).input_ids) - 1

        target[:instruction_len - 1] = IGNORE_TOKEN_ID
        target[instruction_len - 1 + target_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if self.mask_dtype == "bool":
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return dict(
            input_ids=input_ids,
            labels=target,
            attention_mask=attention_mask,
        )

    def preprocess_by_dialog(self, source) -> Dict:
        conv = get_conv_template(self.template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if "instruction" in source:
            conv.set_system_message(source["instruction"])

        source_convs = source["conversations"]
        if roles[source_convs[0]["from"]] != conv.roles[0]:

            source_convs = source_convs[1:]

        conv.messages = []
        for j, sentence in enumerate(source_convs):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{source}"
            message = sentence["value"]
            if message == '':
                message = '<none>'
            conv.append_message(role, message)

        prompt = conv.get_prompt()

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids[0]
        target = input_ids.clone()

        conversation = self.tokenizer.decode(input_ids)

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
        sep = conv.sep + conv.roles[1] + ": "
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        for i, turn in enumerate(turns):
            if turn == "":
                break

            turn += conv.sep2

            turn_len = len(self.tokenizer(turn).input_ids) - 1

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            parts[0] = parts[0].rstrip()

            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len - 1] = IGNORE_TOKEN_ID
            cur_len += turn_len - 1

        if cur_len < self.max_length:
            target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if self.mask_dtype == "bool":
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return dict(
            input_ids=input_ids,
            labels=target,
            attention_mask=attention_mask,
        )

    def preprocess(self, source) -> Dict:

        if _is_world_model_sample(source):
            return self.preprocess_world_model(source)

        if self.template_name == "alpaca":
            return self.preprocess_by_round(source)
        else:
            return self.preprocess_by_dialog(source)

class JsonSupervisedDataset(Dataset):

    def __init__(self, data_path: str, preprocessor: Preprocessor):
        self.samples = self._load_raw_samples(data_path)
        self._processed = [preprocessor.preprocess(sample) for sample in self.samples]

    @staticmethod
    def _load_raw_samples(data_path: str) -> List[Dict]:
        def _load_entire_file():
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            data = _load_entire_file()
        except json.JSONDecodeError:
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        if isinstance(data, dict):
            for key in ("data", "train"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError(f"Unsupported JSON structure in {data_path}")

        if not isinstance(data, list):
            raise ValueError(f"Unsupported JSON structure in {data_path}")

        return data

    def __len__(self):
        return len(self._processed)

    def __getitem__(self, idx):
        return self._processed[idx]

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:

    dataset_cls = Preprocessor(tokenizer=tokenizer,
                               max_length=data_args.cutoff_len,
                               template_name=data_args.template_name,
                               mask_dtype=data_args.mask_dtype)

    rank0_print(f"Loading training data from {data_args.data_path}")
    cache_dir = os.path.join(data_args.cache_path, "caches_base")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_files = {"train": data_args.data_path}

    if load_dataset is None:
        rank0_print("datasets package not found, falling back to local JSON loader.")
        train_dataset = JsonSupervisedDataset(data_args.data_path, dataset_cls)
        eval_dataset = (
            JsonSupervisedDataset(data_args.eval_data_path, dataset_cls)
            if data_args.eval_data_path
            else None
        )
    else:
        if data_args.eval_data_path:
            rank0_print(f"Loading eval data from {data_args.eval_data_path}")
            data_files["eval"] = data_args.eval_data_path

            dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

            train_dataset = dataset["train"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
            eval_dataset = dataset["eval"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
        else:
            dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

            train_dataset = dataset["train"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
            eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
