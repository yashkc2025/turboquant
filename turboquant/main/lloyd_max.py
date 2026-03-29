import numpy as np
from scipy import integrate

def gaussian_pdf(x, d):
    sigma = 1.0 / np.sqrt(d)
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def lloyd_max(d: int, b: int, n_iter: int = 100):
    n_levels = 2 ** b
    sigma = 1.0 / np.sqrt(d)
    lo, hi = -5 * sigma, 5 * sigma

    centroids = np.linspace(lo, hi, n_levels)

    for _ in range(n_iter):
        lower = np.array([lo])
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        upper = np.array([hi])

        bounds = np.concatenate([lower, midpoints, upper])
        new_c = np.zeros_like(centroids)

        for i in range(n_levels):
            a, b_ = bounds[i], bounds[i+1]
            num, _ = integrate.quad(lambda x: x * gaussian_pdf(x, d), a, b_)
            den, _ = integrate.quad(lambda x: gaussian_pdf(x, d), a, b_)
            new_c[i] = num / den if den > 1e-12 else centroids[i]

        difference = new_c - centroids          # element-wise difference
        abs_difference = np.abs(difference)     # magnitude of movement
        max_change = np.max(abs_difference)     # largest movement

        # If movement is very small, we consider it converged
        tolerance = 1e-6
        if max_change < tolerance:
            break
        centroids = new_c

    return np.sort(centroids)