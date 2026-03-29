import numpy as np

class QJL:
    def __init__(self, d: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((d, d)).astype(np.float32)
        self.d = d

    def quantize(self, r: np.ndarray):
        return np.sign(r @ self.S.T).astype(np.int8)
    
    def dequantize(self, z: np.ndarray, gamma: np.ndarray):
        projected = z @ self.S
        scaling_constant = np.sqrt(np.pi / 2) / self.d
        scaled_projection = projected * scaling_constant

        # Expand gamma to match dimensions for broadcasting
        result = scaled_projection * gamma[..., None]

        return result