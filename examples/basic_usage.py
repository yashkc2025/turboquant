import numpy as np
from turboquant.main.prod import TurboQuantProd

def usage_example():
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  HOW TO USE: Simple API")
    print(sep)

    rng = np.random.default_rng(99)
    d = 128

    # Your embedding vectors (must be unit-norm — call normalize_batch if not)
    embeddings = rng.standard_normal((1000, d))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    query = rng.standard_normal(d)
    query /= np.linalg.norm(query)

    print("""
  # ── Scenario A: Compress a database for nearest-neighbour search ──
  tq = TurboQuantProd(d=128, b=4)          # 4-bit, unbiased IP estimator
  idx, qjl, gamma = tq.quantize(embeddings) # compress once, store on disk
  # ... later at query time ...
  db_hat = tq.dequantize(idx, qjl, gamma)   # reconstruct (~4× smaller)
  scores  = query @ db_hat.T                # standard inner product
  topk    = np.argsort(-scores)[:10]        # retrieve top-10

  # ── Scenario B: KV-cache (streaming, one vector at a time) ──
  tq = TurboQuantMSE(d=128, b=3)           # 3-bit, MSE-optimal
  for key_vec in stream_of_keys:
      idx = tq.quantize(key_vec[np.newaxis]) # quantize immediately
      store(idx)                             # only store 3 bits/coord
  # At attention time:
  k_hat = tq.dequantize(load(idx))

  # ── Scenario C: Non-integer bitwidth (2.5-bit) ──
  tqf = TurboQuantFractional(d=128, target_b=2.5, calibration_data=embeddings[:50])
  idx_hi, idx_lo = tqf.quantize(embeddings)
  db_hat = tqf.dequantize(idx_hi, idx_lo)   # 6.4× compression vs fp16
    """)

    # Actually run scenario A
    print("  Running scenario A in code...")
    tq = TurboQuantProd(d=d, b=4)
    idx, qjl, gamma = tq.quantize(embeddings)
    db_hat = tq.dequantize(idx, qjl, gamma)
    scores = query @ db_hat.T
    topk = np.argsort(-scores)[:5]
    exact = np.argsort(-(query @ embeddings.T))[:5]
    overlap = len(set(topk) & set(exact))
    print(f"  Top-5 recall: {overlap}/5 correct  (exact={list(exact)}, approx={list(topk)})")
    print(f"  Memory: {embeddings.nbytes/1024:.0f} KB → "
          f"{(idx.nbytes + qjl.nbytes + gamma.nbytes)/1024:.0f} KB  "
          f"({embeddings.nbytes / (idx.nbytes+qjl.nbytes+gamma.nbytes):.1f}× compression)")

if __name__ == "__main__":
    usage_example()