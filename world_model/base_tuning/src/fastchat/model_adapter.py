import os
import sys
from typing import Dict, List, Optional
import warnings

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
from conversation import Conversation, get_conv_template
from compression import load_compress_model_and_tokenizer

class BaseModelAdapter:

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        if from_pretrained_kwargs["load_in_8bit"]:
            model, tokenizer = load_compress_model_and_tokenizer(
                model_path=model_path,
                device=from_pretrained_kwargs["device"],
                torch_dtype=from_pretrained_kwargs["torch_dtype"],
                use_fast=self.use_fast_tokenizer,
                revision=revision,
            )
            return model, tokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
                legacy=False,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True, legacy=False,
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

model_adapters: List[BaseModelAdapter] = []

def register_model_adapter(cls):
    model_adapters.append(cls())

@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")

def get_conversation_template(model_path: str) -> Conversation:
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

def remove_parent_directory_name(model_path):
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]

class VicunaAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision, legacy=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        self.raise_warning_for_old_weights(model)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.3: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )

class AiroborosAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "airoboros" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("airoboros_v1")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if "mpt" not in model_path.lower():
            return super().load_model(model_path, from_pretrained_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=8192,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        return model, tokenizer

class CodeT5pAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "codet5p" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

class T5Adapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "t5" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = T5Tokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

class KoalaAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "koala" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("koala_v1")

class AlpacaAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "alpaca" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")

class ChatGLMAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "chatglm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        return get_conv_template("chatglm")

class DollyV2Adapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "dolly-v2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )

        tokenizer.eos_token_id = 50277
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolly_v2")

class OasstPythiaAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "oasst" in model_path and "pythia" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_pythia")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

class OasstLLaMAAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        if "openassistant-sft-7-llama-30b-hf" in model_path:
            return True
        return "oasst" in model_path and "pythia" not in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_llama")

class PythiaAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "pythia" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

class StableLMAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "stablelm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stablelm")

class MPTAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "mpt" in model_path and not "airoboros" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=8192,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "mpt-7b-chat" in model_path:
            return get_conv_template("mpt-7b-chat")
        elif "mpt-30b-chat" in model_path:
            return get_conv_template("mpt-30b-chat")
        elif "mpt-30b-instruct" in model_path:
            return get_conv_template("mpt-30b-instruct")
        else:
            print(
                "Warning: Loading base MPT model with `zero_shot` conversation configuration.  "
                "If this is not desired, inspect model configurations and names."
            )
            return get_conv_template("zero_shot")

class BaizeAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "baize" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("baize")

class RwkvAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "rwkv-4" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from fastchat.model.rwkv_model import RwkvModel

        model = RwkvModel(model_path)
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", revision=revision
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("rwkv")

class OpenBuddyAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "openbuddy" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openbuddy")

class PhoenixAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "phoenix" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phoenix")

class ReaLMAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "ReaLM" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ReaLM-7b-v1")

class ChatGPTAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return model_path in ("gpt-3.5-turbo", "gpt-4")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")

class ClaudeAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return model_path in ["claude-2", "claude-instant-1"]

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("claude")

class BardAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return model_path == "bard"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")

class PaLM2Adapter(BaseModelAdapter):

    def match(self, model_path: str):
        return model_path == "palm-2"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")

class BiLLaAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "billa" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("billa")

class RedPajamaINCITEAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "redpajama-incite" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("redpajama-incite")

class H2OGPTAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "h2ogpt" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("h2ogpt")

class RobinAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "robin" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("Robin")

class SnoozyAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "gpt4all" in model_path and "snoozy" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("snoozy")

class WizardLMAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardlm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "13b" in model_path or "30b" in model_path or "70b" in model_path:
            return get_conv_template("vicuna_v1.1")
        else:

            return get_conv_template("one_shot")

class ManticoreAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "manticore" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("manticore")

class GuanacoAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "guanaco" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )

        tokenizer.eos_token_id = model.config.eos_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zero_shot")

class ChangGPTAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "polyglot" in model_path and "chang" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")

class CamelAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "camel" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")

class TuluAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "tulu" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tulu")

class FalconAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "falcon" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")

        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )

        tokenizer.pad_token_id = 9
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("falcon")

class TigerBotAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "tigerbot" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tigerbot")

class BaichuanAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "baichuan" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:

        if "chat" in model_path.lower():
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")

class XGenAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "xgen" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model.config.eos_token_id = 50256
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("xgen")

class NousHermesAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "nous-hermes" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")

class InternLMChatAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "internlm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        model = model.eval()
        if "8k" in model_path.lower():
            model.config.max_sequence_length = 8192
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("internlm-chat")

class StarChatAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "starchat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("starchat")

class Llama2Adapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "llama-2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

class CuteGPTAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "cutegpt" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<end>")
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cutegpt")

class OpenOrcaAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "openorca" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("open-orca")

class WizardCoderAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardcoder" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:

        return get_conv_template("alpaca")

class QwenChatAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "qwen" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from transformers.generation import GenerationConfig

        revision = from_pretrained_kwargs.get("revision", "main")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        config.fp16 = True
        generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        ).eval()
        if hasattr(model.config, "use_dynamic_ntk") and model.config.use_dynamic_ntk:
            model.config.max_sequence_length = 16384
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.pad_token_id = generation_config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-7b-chat")

class BGEAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "bge" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModel.from_pretrained(
            model_path,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        if hasattr(model.config, "max_position_embeddings") and hasattr(
            tokenizer, "model_max_length"
        ):
            model.config.max_sequence_length = min(
                model.config.max_position_embeddings, tokenizer.model_max_length
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

class E5Adapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "e5-" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModel.from_pretrained(
            model_path,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        if hasattr(model.config, "max_position_embeddings") and hasattr(
            tokenizer, "model_max_length"
        ):
            model.config.max_sequence_length = min(
                model.config.max_position_embeddings, tokenizer.model_max_length
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

class AquilaChatAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "aquila" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("aquila-chat")

class Lamma2ChineseAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "llama2-chinese" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama2-chinese")

class VigogneInstructAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vigogne" in model_path.lower() and "instruct" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")

class VigogneChatAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vigogne" in model_path.lower() and "chat" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vigogne-chat")

class OpenLLaMaOpenInstructAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return (
            "open-llama" in model_path.lower() and "open-instruct" in model_path.lower()
        )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")

class CodeLlamaAdapter(BaseModelAdapter):

    def match(self, model_path: str):
        return "codellama" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

register_model_adapter(VicunaAdapter)
register_model_adapter(AiroborosAdapter)
register_model_adapter(CodeT5pAdapter)
register_model_adapter(T5Adapter)
register_model_adapter(KoalaAdapter)
register_model_adapter(AlpacaAdapter)
register_model_adapter(ChatGLMAdapter)
register_model_adapter(DollyV2Adapter)
register_model_adapter(OasstPythiaAdapter)
register_model_adapter(OasstLLaMAAdapter)
register_model_adapter(StableLMAdapter)
register_model_adapter(BaizeAdapter)
register_model_adapter(RwkvAdapter)
register_model_adapter(OpenBuddyAdapter)
register_model_adapter(PhoenixAdapter)
register_model_adapter(BardAdapter)
register_model_adapter(PaLM2Adapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(ClaudeAdapter)
register_model_adapter(MPTAdapter)
register_model_adapter(BiLLaAdapter)
register_model_adapter(RedPajamaINCITEAdapter)
register_model_adapter(H2OGPTAdapter)
register_model_adapter(RobinAdapter)
register_model_adapter(SnoozyAdapter)
register_model_adapter(WizardLMAdapter)
register_model_adapter(ManticoreAdapter)
register_model_adapter(GuanacoAdapter)
register_model_adapter(CamelAdapter)
register_model_adapter(ChangGPTAdapter)
register_model_adapter(TuluAdapter)
register_model_adapter(FalconAdapter)
register_model_adapter(TigerBotAdapter)
register_model_adapter(BaichuanAdapter)
register_model_adapter(XGenAdapter)
register_model_adapter(NousHermesAdapter)
register_model_adapter(PythiaAdapter)
register_model_adapter(InternLMChatAdapter)
register_model_adapter(StarChatAdapter)
register_model_adapter(Llama2Adapter)
register_model_adapter(CuteGPTAdapter)
register_model_adapter(OpenOrcaAdapter)
register_model_adapter(WizardCoderAdapter)
register_model_adapter(QwenChatAdapter)
register_model_adapter(AquilaChatAdapter)
register_model_adapter(BGEAdapter)
register_model_adapter(E5Adapter)
register_model_adapter(Lamma2ChineseAdapter)
register_model_adapter(VigogneInstructAdapter)
register_model_adapter(VigogneChatAdapter)
register_model_adapter(OpenLLaMaOpenInstructAdapter)
register_model_adapter(ReaLMAdapter)
register_model_adapter(CodeLlamaAdapter)

register_model_adapter(BaseModelAdapter)
