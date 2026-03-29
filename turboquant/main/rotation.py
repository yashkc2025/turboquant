import numpy as np

def random_rotation(d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Generate a random matrix with standard normal entries
    random_matrix = rng.standard_normal((d, d))

    # Perform QR decomposition to get an orthogonal matrix
    Q, R = np.linalg.qr(random_matrix)

    # fIX signs to ensure a proper rotation (determinant = +1)
    diagonal_signs = np.sign(np.diag(R))
    sign_matrix = diagonal_signs[np.newaxis, :]

    rotation_matrix = Q * sign_matrix

    return rotation_matrix