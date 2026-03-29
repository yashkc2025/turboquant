import numpy as np
from turboquant.main.mse import TurboQuantMSE
from turboquant.main.qjl import QJL

class TurboQuantProd:
    def __init__(self, d: int, b: int, seed: int = 42):
        self.d = d
        self.b = b
        self.mse = TurboQuantMSE(d, max(1, b-1), seed)
        self.qjl = QJL(d, seed+1)

    def quantize(self, x: np.ndarray):
        indices = self.mse.quantize(x)

        x_approx = self.mse.dequantize(indices)
        residual = x - x_approx
        residual_norm = np.linalg.norm(residual, axis=-1)

        safe_norm = np.maximum(residual_norm, 1e-8)
        residual_direction = residual / np.where(residual_norm[..., np.newaxis] > 0, residual_norm[..., np.newaxis], 1.0)
        direction_code = self.qjl.quantize(residual_direction)

        # Return:
        # - indices from first quantizer
        # - encoded direction of residual
        # - magnitude of residual
        return indices, direction_code, residual_norm

    def dequantize(self, idx, z, gamma):
        return self.mse.dequantize(idx) + self.qjl.dequantize(z, gamma)
    
    def inner_product(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.dequantize(*self.quantize(x)) @ y
    
    def upper_bound(self, y: np.ndarray):
        constant = np.sqrt(3 * np.pi) / 2
        
        total = 0.0
        for value in y:
            total += value * value
        
        normalized = total / self.d
        scaling = 4 ** (-self.b)
        
        return constant * normalized * scaling
