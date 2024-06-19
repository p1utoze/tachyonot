from joblib import Memory
from .config import CACHE_DIR
from .memoize import memoize
mem = Memory(location=CACHE_DIR, verbose=2, compress=5)

__all__ = ["mem", "memoize"]