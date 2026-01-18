from .base import LMAgent

from .fastchat_agent import FastChatAgent

def __getattr__(name):
    if name == "ForesightLocalAgent":
        from .foresight_local_agent import ForesightLocalAgent

        return ForesightLocalAgent
    raise AttributeError(f"module {__name__} has no attribute {name}")
