from turboquant.main.caching import CACHE
from turboquant.main.rotation import random_rotation
import numpy as np

class TurboQuantMSE:
    def __init__(self, dim: int, bits: int, seed: int = 42, verbose: bool = False):
        self.dim = dim
        self.bits = bits

        self._rotation = random_rotation(dim, seed)

        if verbose:
            print(f"[Initializing codebook | dim={dim}, bits={bits}]...", end=" ", flush=True)

        self._centroids = CACHE.get(dim, bits)

        if verbose:
            print("done.")

    def quantize(self, x: np.ndarray) -> np.ndarray:
        rotated = x @ self._rotation.T
        distances = np.abs(rotated[..., np.newaxis] - self._centroids)
        indices = np.argmin(distances, axis=-1)
        return indices.astype(np.uint16)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        return self._centroids[indices] @ self._rotation

    def mse(self, x: np.ndarray):
        reconstructed = self.dequantize(self.quantize(x))
        error = x - reconstructed
        return float(np.mean(np.sum(error**2, axis=-1)))

    # ---- Theoretical bounds (Theorems 1 & 3) ----
    def upper_bound(self):
        return (np.sqrt(3 * np.pi) / 2) * (4 ** -self.bits)

    def lower_bound(self):
        return 4 ** -self.bits