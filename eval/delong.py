import math
import numpy as np

try:
    from scipy.stats import norm as _norm  # optional but preferred
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _zcrit(alpha: float) -> float:
    return float(_norm.ppf(1 - alpha/2)) if _HAS_SCIPY else 1.959963984540054

def _ncdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(_norm.cdf(x))
    # Gaussian CDF fallback via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _pairwise_v(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    # v_{ij} = 1 if pos_i > neg_j; 0.5 if tie; 0 else
    diff = pos[:, None] - neg[None, :]
    return (diff > 0).astype(float) + 0.5 * (diff == 0)

def auc_and_variance(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    v = _pairwise_v(pos, neg)
    auc = float(v.mean())
    v10 = v.mean(axis=1)  # per-positive
    v01 = v.mean(axis=0)  # per-negative
    var = (np.var(v10, ddof=1) / m) + (np.var(v01, ddof=1) / n)
    return auc, float(var)

def auc_delong_ci(y_true, scores, alpha: float = 0.05):
    auc, var = auc_and_variance(y_true, scores)
    if not np.isfinite(var):
        return auc, (float("nan"), float("nan"))
    se = math.sqrt(max(var, 0.0))
    z = _zcrit(alpha)
    return auc, (max(0.0, auc - z * se), min(1.0, auc + z * se))

def delong_2sample(y_true, scores1, scores2, alpha: float = 0.05):
    """
    Paired DeLong for two models on the same (y_true, X_test).
    Returns delta=AUC1-AUC2, its SE, z, p, and a CI for the delta.
    """
    y = np.asarray(y_true).astype(int)
    s1 = np.asarray(scores1, dtype=float)
    s2 = np.asarray(scores2, dtype=float)
    idx_pos, idx_neg = (y == 1), (y == 0)
    pos1, neg1 = s1[idx_pos], s1[idx_neg]
    pos2, neg2 = s2[idx_pos], s2[idx_neg]
    m, n = len(pos1), len(neg1)
    if m == 0 or n == 0:
        return dict(delta=np.nan, se=np.nan, z=np.nan, p=np.nan, ci=(np.nan, np.nan),
                    auc1=np.nan, auc2=np.nan)

    v1 = _pairwise_v(pos1, neg1)
    v2 = _pairwise_v(pos2, neg2)
    auc1 = float(v1.mean()); auc2 = float(v2.mean())

    v10_1, v01_1 = v1.mean(axis=1), v1.mean(axis=0)
    v10_2, v01_2 = v2.mean(axis=1), v2.mean(axis=0)

    var1 = (np.var(v10_1, ddof=1) / m) + (np.var(v01_1, ddof=1) / n)
    var2 = (np.var(v10_2, ddof=1) / m) + (np.var(v01_2, ddof=1) / n)

    cov_v10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] if m > 1 else 0.0
    cov_v01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] if n > 1 else 0.0
    cov12 = (cov_v10 / m) + (cov_v01 / n)

    delta = auc1 - auc2
    var_delta = var1 + var2 - 2.0 * cov12
    se = math.sqrt(max(var_delta, 0.0))
    z = (delta / se) if se > 0 else float("inf")
    p = 2.0 * (1.0 - _ncdf(abs(z)))
    zc = _zcrit(alpha)
    ci = (delta - zc * se, delta + zc * se)
    return dict(delta=float(delta), se=float(se), z=float(z), p=float(p), ci=(float(ci[0]), float(ci[1])),
                auc1=float(auc1), auc2=float(auc2))
