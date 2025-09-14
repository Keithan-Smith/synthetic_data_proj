from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------- safety helpers ----------------
# numeric guards for exp/expm1 and sigmoid
_MAX_EXP32 = 80.0     # exp(88.7) ~ float32 max; keep margin
_MAX_EXP64 = 700.0    # exp(709.) ~ float64 max; keep margin

def _dtype_maxexp(arr: np.ndarray) -> float:
    return _MAX_EXP32 if arr.dtype == np.float32 else _MAX_EXP64

def _safe_exp(x):
    arr = np.asarray(x)
    lim = _dtype_maxexp(arr)
    return np.exp(np.clip(arr, -lim, lim))

def _safe_expm1(x):
    arr = np.asarray(x)
    lim = _dtype_maxexp(arr)
    # avoid large positives; allow moderate negatives for accuracy
    return np.expm1(np.clip(arr, -50.0, lim))

def _sigmoid_stable(x):
    a = np.asarray(x)
    out = np.empty_like(a, dtype=float)
    pos = a >= 0
    # for positive x: 1/(1+exp(-x))
    out[pos] = 1.0 / (1.0 + np.exp(-np.clip(a[pos], None, _dtype_maxexp(a))))
    # for negative x: exp(x)/(1+exp(x))
    en = np.exp(np.clip(a[~pos], -_dtype_maxexp(a), None))
    out[~pos] = en / (1.0 + en)
    # clip away from exact 0/1 for numerical safety
    return np.clip(out, 1e-8, 1 - 1e-8)

# ---------------- schema dataclasses ----------------
@dataclass
class TypeSpec:
    kind: str
    link: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None

@dataclass
class Schema:
    continuous: Dict[str, TypeSpec] = field(default_factory=dict)
    categoricals: List[str] = field(default_factory=list)
    # RankGauss inverse cache
    _rg_mu: Dict[str, float] = field(default_factory=dict, repr=False)
    _rg_sig: Dict[str, float] = field(default_factory=dict, repr=False)
    _rg_qgrid: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)  # 1001-pt quantile grid per col

    def cont_cols(self) -> List[str]:
        return list(self.continuous.keys())

    def cat_cols(self) -> List[str]:
        return list(self.categoricals)

# ---------------- link functions ----------------
def identity(x): return x

def log_link(x):
    return np.log(np.maximum(x, 1e-12))

def inv_log_link(x):
    return _safe_exp(x)

def log1p_link(x):
    return np.log1p(np.maximum(x, 0.0))

def inv_log1p_link(x):
    y = _safe_expm1(x)
    return np.maximum(y, 0.0)

def logit_link(x):
    z = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(z / (1 - z))

def inv_logit_link(x):
    return _sigmoid_stable(x)

# RankGauss helpers
def _rankgauss_forward(s: pd.Series) -> Tuple[np.ndarray, float, float, np.ndarray]:
    r = s.rank(method="average", pct=True).to_numpy(dtype=float)
    eps = 1e-6
    r = np.clip(r, eps, 1 - eps)
    z = norm.ppf(r)
    mu = float(np.nanmean(z))
    sig = float(np.nanstd(z) if np.nanstd(z) > 1e-12 else 1.0)
    zt = (z - mu) / sig
    # cache quantile grid for inverse
    try:
        qgrid = np.quantile(s.to_numpy(dtype=float), np.linspace(0, 1, 1001), method="linear")
    except TypeError:
        # numpy<1.22 fallback
        qgrid = np.quantile(s.to_numpy(dtype=float), np.linspace(0, 1, 1001))
    return zt, mu, sig, qgrid

def _rankgauss_inverse(arr: np.ndarray, mu: float, sig: float, qgrid: np.ndarray) -> np.ndarray:
    z = (arr * sig) + mu
    u = norm.cdf(np.clip(z, -8, 8))  # 0..1
    idx = np.clip((u * 1000).astype(int), 0, 1000)
    return qgrid[idx]

LINKS: Dict[str, Tuple] = {
    "":          (identity, identity),
    "identity":  (identity, identity),
    "none":      (identity, identity),
    "log":       (log_link, inv_log_link),
    "log1p":     (log1p_link, inv_log1p_link),
    "logit":     (logit_link, inv_logit_link),
    # "rankgauss" handled explicitly
}

# ---------------- forward / inverse transforms ----------------
def forward_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c not in out.columns:
            continue
        if spec.link == "rankgauss":
            vals, mu, sig, qgrid = _rankgauss_forward(out[c])
            out[c] = vals
            schema._rg_mu[c] = mu
            schema._rg_sig[c] = sig
            schema._rg_qgrid[c] = qgrid
        else:
            f, _ = LINKS.get(spec.link, LINKS[""])
            out[c] = f(out[c].to_numpy())
    return out

def inverse_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c not in out.columns:
            continue
        if spec.link == "rankgauss":
            mu = schema._rg_mu.get(c, 0.0)
            sig = schema._rg_sig.get(c, 1.0)
            qg = schema._rg_qgrid.get(c, None)
            if qg is None:
                # graceful fallback: return uniform [0,1]
                u = norm.cdf(out[c].to_numpy())
                out[c] = u
            else:
                out[c] = _rankgauss_inverse(out[c].to_numpy(), mu, sig, qg)
        else:
            _, inv = LINKS.get(spec.link, LINKS[""])
            arr = inv(out[c].to_numpy())
            if spec.min_val is not None:
                arr = np.maximum(arr, spec.min_val)
            if spec.max_val is not None:
                arr = np.minimum(arr, spec.max_val)
            out[c] = arr
    return out

# ---------------- inference ----------------
def infer_schema_from_df(
    df: pd.DataFrame,
    max_categorical_card: int = 50,
    transform_overrides: Optional[Dict[str, str]] = None
) -> Schema:
    """
    Heuristically infer which columns are continuous vs categorical and choose link functions.
    Defaults:
      - Probability/ratio in [0,1] → 'logit'
      - Everything else numeric → 'rankgauss'
      - Objects/pandas.Categorical → categorical
    """
    continuous: Dict[str, TypeSpec] = {}
    categoricals: List[str] = []

    for c in df.columns:
        s = df[c]

        # Object / pandas Categorical are categorical
        if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
            categoricals.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            minv = float(np.nanmin(s.to_numpy())) if s.notna().any() else None
            maxv = float(np.nanmax(s.to_numpy())) if s.notna().any() else None

            # choose link
            if minv is not None and maxv is not None and 0.0 <= minv <= 1.0 and maxv <= 1.0:
                link = "logit"
            else:
                link = "rankgauss"

            continuous[c] = TypeSpec(kind="continuous", link=link, min_val=minv, max_val=maxv)
        else:
            categoricals.append(c)

    # apply overrides last
    if transform_overrides:
        for col, ln in transform_overrides.items():
            if col in continuous and (ln in LINKS or ln == "rankgauss"):
                continuous[col].link = ln

    return Schema(continuous=continuous, categoricals=categoricals)
