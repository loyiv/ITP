import logging
from typing import List, Dict, Any, Mapping

logger = logging.getLogger("agent_frame")

class LMAgent:

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")

        self.stop_words = [
            "\nObservation:",
            "\nTask:",
            "\n---",
            "</s>",
            "<|eot_id|>",
            "<|end|>",
            "<|endoftext|>",
        ]

    def __call__(self) -> str:
        pass

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:

        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages
