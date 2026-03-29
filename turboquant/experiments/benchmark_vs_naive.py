import numpy as np
from turboquant.main.mse import TurboQuantMSE
from turboquant.main.prod import TurboQuantProd
from turboquant.misc.simple_quant import NaiveQuant

def run(d: int = 512, n: int = 3000):

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  BENCHMARK: TurboQuant vs Naive Uniform Quantizer")
    print(f"  d={d}, n={n} random unit-norm vectors")
    print(sep)

    rng = np.random.default_rng(0)

    # make random data + normalize
    X = rng.standard_normal((n, d)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # random query vector
    y = rng.standard_normal(d)
    y /= np.linalg.norm(y)

    # true inner products (ground truth)
    true_ip = X @ y

    # numbers from paper for reference
    paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    print(f"\n  {'b':>3}  {'TurboQ MSE':>12}  {'Naive MSE':>12}  {'Improvement':>13}  "
          f"{'Paper ref':>10}  {'TurboQ IP bias':>15}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*13}  {'─'*10}  {'─'*15}")

    for b in [1, 2, 3, 4]:
        # set up quantizers
        tq = TurboQuantMSE(dim=d, bits=b, seed=42)
        tq_p = TurboQuantProd(d=d, b=b, seed=42)
        nq   = NaiveQuant(d=d, b=b)

        # reconstruction error
        turbo_mse = tq.mse(X)
        naive_mse = nq.mse(X)

        # how much better turbo is
        ratio = naive_mse / turbo_mse

        # inner product estimate + bias
        ip_est  = tq_p.inner_product(X, y)
        ip_bias = float(np.mean(ip_est - true_ip))

        print(f"  {b:>3}  {turbo_mse:>12.5f}  {naive_mse:>12.5f}  "
              f"  {ratio:>8.2f}×       {paper_mse[b]:>10.3f}  {ip_bias:>15.6f}")

    print(f"\n  → improvement = how many times turbo beats naive")
    print(f"  → IP bias should be ~0")

if __name__ == "__main__":
    run()