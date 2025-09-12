import math
import numpy as np
from typing import Dict

def _psi_matrix(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """
    Pairwise comparison matrix ψ(x_i, y_j) for positives x and negatives y:
      1.0 if x_i > y_j
      0.5 if x_i == y_j
      0.0 if x_i < y_j
    Returns shape (m, n) where m=#pos, n=#neg.
    """
    diff = pos[:, None] - neg[None, :]
    res = (diff > 0).astype(float)
    res[diff == 0] = 0.5
    return res

def _v10_v01(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute DeLong per-positive (V10) and per-negative (V01) contributions.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    pos = s[y == 1]
    neg = s[y == 0]
    m = pos.size
    n = neg.size
    if m == 0 or n == 0:
        return np.array([]), np.array([]), np.nan

    psi = _psi_matrix(pos, neg)        # (m, n)
    v10 = psi.mean(axis=1)             # length m
    v01 = psi.mean(axis=0)             # length n
    auc = v10.mean()
    return v10, v01, auc

def _cov(x: np.ndarray, y: np.ndarray) -> float:
    """Sample covariance with ddof=1; returns 0 if too short."""
    if x.size <= 1 or y.size <= 1:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1])

def _var(x: np.ndarray) -> float:
    """Sample variance with ddof=1; returns 0 if too short."""
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))

def _normal_cdf(z: float) -> float:
    """Φ(z) without SciPy using erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def delong_roc_test(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> Dict[str, float]:
    """
    Two-model DeLong test for AUC difference on the **same** labels y_true.
    Returns dict with: auc_a, auc_b, delta, z, p_value, var_a, var_b, cov_ab.
    """
    y = np.asarray(y_true).astype(int)
    pa = np.asarray(p_a).astype(float)
    pb = np.asarray(p_b).astype(float)

    # sanity: same population
    if y.shape[0] != pa.shape[0] or y.shape[0] != pb.shape[0]:
        raise ValueError("y_true, p_a, p_b must have the same length")

    v10_a, v01_a, auc_a = _v10_v01(y, pa)
    v10_b, v01_b, auc_b = _v10_v01(y, pb)

    if np.isnan(auc_a) or np.isnan(auc_b):
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "delta": float("nan"), "z": float("nan"), "p_value": float("nan"),
                "var_a": float("nan"), "var_b": float("nan"), "cov_ab": float("nan")}

    m = (y == 1).sum()
    n = (y == 0).sum()

    # Variances per DeLong
    var_a = _var(v10_a) / max(m, 1) + _var(v01_a) / max(n, 1)
    var_b = _var(v10_b) / max(m, 1) + _var(v01_b) / max(n, 1)

    # Covariance between AUC_A and AUC_B
    cov_ab = _cov(v10_a, v10_b) / max(m, 1) + _cov(v01_a, v01_b) / max(n, 1)

    delta = float(auc_a - auc_b)
    var_delta = max(var_a + var_b - 2.0 * cov_ab, 1e-18)  # guard
    z = delta / math.sqrt(var_delta)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))  # two-sided

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta": float(delta),
        "z": float(z),
        "p_value": float(p),
        "var_a": float(var_a),
        "var_b": float(var_b),
        "cov_ab": float(cov_ab),
    }
