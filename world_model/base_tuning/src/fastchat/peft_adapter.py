import os
from peft import PeftConfig, PeftModel
from model_adapter import get_model_adapter
from conversation import Conversation

peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)

peft_model_cache = {}

class PeftModelAdapter:

    name = "PeftModelAdapter"

    def match(self, model_path: str):
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            return True
        return "peft" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):

        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        if "peft" in base_model_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )

        if peft_share_base_weights:
            if base_model_path in peft_model_cache:
                model, tokenizer = peft_model_cache[base_model_path]

                model.load_adapter(model_path, adapter_name=model_path)
            else:
                base_adapter = get_model_adapter(base_model_path)
                base_model, tokenizer = base_adapter.load_model(
                    base_model_path, from_pretrained_kwargs
                )

                model = PeftModel.from_pretrained(
                    base_model, model_path, adapter_name=model_path
                )
                peft_model_cache[base_model_path] = (model, tokenizer)
            return model, tokenizer

        base_adapter = get_model_adapter(base_model_path)
        base_model, tokenizer = base_adapter.load_model(
            base_model_path, from_pretrained_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        config = PeftConfig.from_pretrained(model_path)
        if "peft" in config.base_model_name_or_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )

        base_model_path = config.base_model_name_or_path
        base_adapter = get_model_adapter(base_model_path)
        return base_adapter.get_default_conv_template(config.base_model_name_or_path)
