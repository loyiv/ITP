from typing import Any

def build_arg_parser():
    from .runner import build_arg_parser as _build
    return _build()

def evaluate_from_args(args: Any) -> None:
    from .runner import evaluate_from_args as _eval
    return _eval(args)

class ForesightEvaluator:
    def __new__(cls, *args, **kwargs):
        from .runner import ForesightEvaluator as _Cls
        return _Cls(*args, **kwargs)

__all__ = ["ForesightEvaluator", "build_arg_parser", "evaluate_from_args"]

