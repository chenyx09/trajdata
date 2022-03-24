from pathlib import Path


class BaseCache:
    def __init__(self, cache_location: str) -> None:
        # Ensuring the specified cache folder exists
        self.path = Path(cache_location).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
