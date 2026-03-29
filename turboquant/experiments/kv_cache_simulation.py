import numpy as np
from turboquant.main.prod import TurboQuantProd
from turboquant.misc.simple_quant import NaiveQuant

def demo_kv_cache(seq_len: int = 512, d: int = 128, n_heads: int = 8):

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  DEMO 2: KV-Cache Compression")
    print(f"  seq_len={seq_len}, d={d}, n_heads={n_heads}")
    print(f"  NaiveQuant uses the theoretical unit-sphere range (no data peeking).")
    print(sep)

    rng = np.random.default_rng(2)

    K = rng.standard_normal((n_heads, seq_len, d)).astype(np.float64)
    K /= np.linalg.norm(K, axis=-1, keepdims=True)
    Q_vec = rng.standard_normal((n_heads, 1, d)).astype(np.float64)
    Q_vec /= np.linalg.norm(Q_vec, axis=-1, keepdims=True)

    print(f"\n  Memory for full KV cache (fp16): {n_heads * seq_len * d * 2 / 1024:.1f} KB")
    print(f"\n  {'Config':35s}  {'Attn score MSE':>16}  {'Memory':>10}  {'Compression':>13}")
    print(f"  {'─'*35}  {'─'*16}  {'─'*10}  {'─'*13}")

    attn_exact = (Q_vec @ K.transpose(0, 2, 1)).squeeze(1)

    configs = [
        ("Full precision (fp16)", None, None),
        ("TurboQuant 4-bit", "turbo", 4),
        ("TurboQuant 2-bit", "turbo", 2),
        ("Naive 4-bit (fixed range, no peeking)", "naive", 4),
        ("Naive 2-bit (fixed range, no peeking)", "naive", 2),
    ]

    for label, mode, b in configs:
        if mode is None:
            mem_kb = n_heads * seq_len * d * 2 / 1024
            print(f"  {label:35s}  {'— (reference)':>16}  {mem_kb:>9.1f}K  {'1.0×':>13}")
            continue

        total_mse = 0.0
        for h in range(n_heads):
            Kh = K[h]
            Qh = Q_vec[h, 0]

            if mode == "turbo":
                tq = TurboQuantProd(d=d, b=b, seed=h)
                K_hat = tq.dequantize(*tq.quantize(Kh))
            elif mode == "naive":
                # FIX: use theoretical unit-sphere range — no peeking at Kh
                nq = NaiveQuant(d=d, b=b)
                sigma = 1.0 / np.sqrt(d)
                nq._lo, nq._hi = -3 * sigma, 3 * sigma
                n_lev = 2 ** b
                nq._edges = np.linspace(nq._lo, nq._hi, n_lev + 1)
                nq._centers = 0.5 * (nq._edges[:-1] + nq._edges[1:])
                K_hat = nq.dequantize(nq.quantize(Kh))

            total_mse += float(np.mean((K_hat @ Qh - K[h] @ Qh) ** 2))

        avg_mse = total_mse / n_heads
        mem_kb = n_heads * seq_len * d * b / 8 / 1024
        print(f"  {label:35s}  {avg_mse:>16.6f}  {mem_kb:>9.1f}K  {16/b:>12.1f}×")

    print("""
  Note: on ideal Gaussian unit-sphere keys, a uniform grid with the correct
  range is competitive at 4-bit. TurboQuant's advantage is:
    1. No calibration needed — works on any distribution out of the box.
    2. Provably beats naive at 2-bit due to the Lloyd-Max codebook shape.
    3. Guaranteed worst-case bounds (Theorem 1) — naive has no such guarantee.
    """)

if __name__ == "__main__":
    demo_kv_cache()
