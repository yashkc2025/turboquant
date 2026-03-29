import numpy as np

class NaiveQuant:
    """
    very basic uniform quantizer
    just clips values and splits into equal bins
    """

    def __init__(self, d: int, b: int, clip_sigma: float = 3.0):
        self.d = d
        self.b = b
        self.clip_sigma = clip_sigma

        self._lo = None
        self._hi = None

    def _fit(self, x: np.ndarray):
        std = float(np.std(x))

        # pick range based on std
        self._lo = -self.clip_sigma * std
        self._hi = self.clip_sigma * std

        # number of bins
        n_bins = 2 ** self.b

        # edges and centers
        self._edges = np.linspace(self._lo, self._hi, n_bins + 1)
        self._centers = 0.5 * (self._edges[:-1] + self._edges[1:])

    def quantize(self, x: np.ndarray) -> np.ndarray:
        if self._lo is None:
            self._fit(x)

        # clip then find which bin it falls into
        clipped = np.clip(x, self._lo, self._hi)
        idx = np.searchsorted(self._edges[1:-1], clipped)

        return idx.astype(np.uint16)

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        # map back to bin centers
        return self._centers[idx]

    def mse(self, x: np.ndarray) -> float:
        # simple reconstruction error
        x_hat = self.dequantize(self.quantize(x))
        err = x - x_hat
        return float(np.mean(np.sum(err**2, axis=-1)))