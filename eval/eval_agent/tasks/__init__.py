from .base import Task
from .alfworld import AlfWorldTask

__all__ = [
    "Task",
    "AlfWorldTask",
    "SciWorldTask",
]

def __getattr__(name: str):

    if name == "SciWorldTask":
        try:
            from .sciworld import SciWorldTask

            return SciWorldTask
        except Exception as e:
            raise ImportError(
                "SciWorldTask 依赖额外包 `scienceworld`（以及 SciWorld 运行环境）。"
                "当前环境未安装/不可用，因此无法导入 SciWorldTask。"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
