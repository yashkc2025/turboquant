import numpy as np
from turboquant.main.lloyd_max import lloyd_max

class Caching:
    def __init__(self):
        self._cache = {}

    def get(self, d: int, b: int):
        key = (d, b)
        if key not in self._cache:
            self._cache[key] = lloyd_max(d, b)
        return self._cache[key]


# Global shared cache
CACHE = Caching()