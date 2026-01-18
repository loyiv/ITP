import functools

@functools.lru_cache(maxsize=128)
def load_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        content = f.read()
    return content
