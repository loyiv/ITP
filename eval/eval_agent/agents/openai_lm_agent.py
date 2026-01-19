import openai
import logging
import backoff
import os

from .base import LMAgent

logger = logging.getLogger("agent_frame")

class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if "api_base" in config:
            openai.api_base = config['api_base']

        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key. Please set env var: {api_key_env}")
        openai.api_key = api_key

    @backoff.on_exception(
        backoff.fibo,

        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )
    def __call__(self, messages) -> str:

        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        return response.choices[0].message["content"]
