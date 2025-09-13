from __future__ import annotations
import math
import argparse
import numpy as np
import pandas as pd

def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (1..N) with tie handling."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = Z.size
    T = np.empty(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        # average of ranks i..(j-1), +1 to convert 0-based to 1-based
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    R = np.empty(N, dtype=float)
    R[J] = T
    return R

def _fast_delong(scores: np.ndarray, y: np.ndarray):
    """
    Vectorized DeLong for one or more models.
    scores: shape (K, N) with higher=more positive
    y:      shape (N,), 1=positive, 0=negative
    Returns: aucs (K,), covariance matrix (K,K)
    """
    y = y.astype(int)
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    assert m > 0 and n > 0, "Need both positive and negative samples."

    K, N = scores.shape
    v10 = np.empty((K, m), dtype=float)
    v01 = np.empty((K, n), dtype=float)
    aucs = np.empty(K, dtype=float)

    for k in range(K):
        s = scores[k]
        A = s[pos]
        B = s[neg]
        r = _midrank(np.concatenate([A, B], axis=0))
        rA = r[:m]
        rB = r[m:]
        v10[k] = (rA - (m + 1) / 2.0) / n
        v01[k] = ((m + n + 1) / 2.0 - rB) / m
        aucs[k] = v10[k].mean()

    # population covariances
    S10 = np.cov(v10, bias=True)
    S01 = np.cov(v01, bias=True)
    cov = S10 / m + S01 / n
    return aucs, cov

def _norm_cdf(z: float) -> float:
    # stable standard normal CDF without SciPy
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def delong_two_model_test(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray):
    """
    Returns dict with AUCs, delta, z, p, se, ci95.
    """
    scores = np.vstack([s_a, s_b])
    aucs, cov = _fast_delong(scores, y)
    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = math.sqrt(max(var, 1e-18))
    z = delta / se
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    z95 = 1.959963984540054
    ci = (delta - z95 * se, delta + z95 * se)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": float(delta),
        "se": float(se),
        "z": float(z),
        "p": float(p),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }

def _read_scores_csv(path: str):
    # Expect columns: 'y' and 'score'
    df = pd.read_csv(path)
    if "y" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path} must contain columns: y, score")
    y = df["y"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()
    return y, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_a", required=True, help="CSV with columns y,score (run A)")
    ap.add_argument("--scores_b", required=True, help="CSV with columns y,score (run B)")
    args = ap.parse_args()

    y_a, s_a = _read_scores_csv(args.scores_a)
    y_b, s_b = _read_scores_csv(args.scores_b)
    if len(y_a) != len(y_b) or np.any(y_a != y_b):
        raise ValueError("Label vectors must match (same test set and order).")
    res = delong_two_model_test(y_a, s_a, s_b)
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
