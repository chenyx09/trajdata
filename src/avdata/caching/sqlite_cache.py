from avdata.caching.base_cache import BaseCache


class SQLiteCache(BaseCache):
    def __init__(self, cache_location: str) -> None:
        super().__init__(cache_location)
