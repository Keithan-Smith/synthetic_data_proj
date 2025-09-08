from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

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

def _fit_rankgauss_quantiles(s: pd.Series, ngrid: int = 2001) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").dropna().values
    if len(v) < 10:
        # fallback grid to keep invertible behavior
        return np.linspace(np.nanmin(v) if len(v) else -1.0, np.nanmax(v) if len(v) else 1.0, ngrid)
    return np.quantile(v, np.linspace(0, 1, ngrid))

def _rank_to_norm(u: np.ndarray) -> np.ndarray:
    # map u in (0,1) to N(0,1) via inverse error function
    u = np.clip(u, 1e-8, 1-1e-8)
    return np.sqrt(2.0) * np.erfinv(2*u - 1.0)

def _norm_to_rank(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))

def _rankgauss_forward(x: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    # empirical CDF via quantile grid
    idx = np.searchsorted(qgrid, x, side="left")
    u = idx / max(len(qgrid) - 1, 1)
    return _rank_to_norm(u)

def _rankgauss_inverse(z: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    u = _norm_to_rank(z)
    idx = (u * (len(qgrid) - 1)).astype(int)
    idx = np.clip(idx, 0, len(qgrid) - 1)
    return qgrid[idx]

# ---------------- schema dataclasses ----------------
@dataclass
class TypeSpec:
    kind: str
    link: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Schema:
    continuous: Dict[str, TypeSpec] = field(default_factory=dict)
    categoricals: List[str] = field(default_factory=list)

    def cont_cols(self) -> List[str]:
        return list(self.continuous.keys())

    def cat_cols(self) -> List[str]:
        return list(self.categoricals)

# ---------------- link functions ----------------
def identity(x): return x

def log_link(x):
    # protect against log(0) and negatives with a tiny floor
    return np.log(np.maximum(x, 1e-12))

def inv_log_link(x):
    # overflow-safe exp
    return _safe_exp(x)

def log1p_link(x):
    # log(1+x), floor at 0 so negatives don’t slip through
    return np.log1p(np.maximum(x, 0.0))

def inv_log1p_link(x):
    # overflow-safe expm1
    y = _safe_expm1(x)
    # ensure non-negative domain after inverse
    return np.maximum(y, 0.0)

def logit_link(x):
    z = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(z / (1 - z))

def inv_logit_link(x):
    return _sigmoid_stable(x)

LINKS: Dict[str, Tuple] = {
    "":          (identity, identity),
    "identity":  (identity, identity),
    "none":      (identity, identity),
    "log":       (log_link, inv_log_link),
    "log1p":     (log1p_link, inv_log1p_link),
    "logit":     (logit_link, inv_logit_link),
}

# ---------------- forward / inverse transforms ----------------
def forward_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c not in out.columns:
            continue
        arr = pd.to_numeric(out[c], errors="coerce").to_numpy()
        link = (spec.link or "").lower()

        if link == "rankgauss":
            # fit grid once per column and cache in spec.params
            if "qgrid" not in spec.params or spec.params.get("qgrid") is None:
                spec.params["qgrid"] = _fit_rankgauss_quantiles(out[c])
            out[c] = _rankgauss_forward(arr, spec.params["qgrid"])
        else:
            f, _ = LINKS.get(link, LINKS[""])
            out[c] = f(arr)
    return out

def inverse_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c not in out.columns:
            continue
        arr = pd.to_numeric(out[c], errors="coerce").to_numpy()
        link = (spec.link or "").lower()

        if link == "rankgauss":
            qgrid = spec.params.get("qgrid")
            if qgrid is None:
                # no grid: identity fallback
                inv_arr = arr
            else:
                inv_arr = _rankgauss_inverse(arr, qgrid)
        else:
            _, inv = LINKS.get(link, LINKS[""])
            inv_arr = inv(arr)

        if spec.min_val is not None:
            inv_arr = np.maximum(inv_arr, spec.min_val)
        if spec.max_val is not None:
            inv_arr = np.minimum(inv_arr, spec.max_val)
        out[c] = inv_arr
    return out

# ---------------- inference ----------------
def infer_schema_from_df(
    df: pd.DataFrame,
    max_categorical_card: int = 50,
    transform_overrides: Optional[Dict[str, str]] = None,
    prefer_rankgauss = True
) -> Schema:
    """
    Heuristically infer which columns are continuous vs categorical and choose link functions.
    - Integers/floats are 'continuous' by default, unless they look like high-cardinality codes you want categorical.
    - If a column is in [0,1], prefer 'logit'.
    - If nonnegative and roughly multiplicative spread, prefer 'log' (or override to 'log1p' via transform_overrides).
    - transform_overrides: dict col -> {"log","log1p","logit","",...}, applied last.
    """
    continuous: Dict[str, TypeSpec] = {}
    categoricals: List[str] = []

    for c in df.columns:
        s = df[c]

        # Treat pandas Categorical and object as categorical outright
        if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
            categoricals.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            # Decide continuous vs categorical: keep numeric as continuous by default.
            # If you want small-cardinality ints as categorical, handle upstream (as you already do).
            minv = float(np.nanmin(s.to_numpy())) if s.notna().any() else None
            maxv = float(np.nanmax(s.to_numpy())) if s.notna().any() else None

            # choose link
            link = ""
            if minv is not None and maxv is not None and 0.0 <= minv <= 1.0 and maxv <= 1.0:
                link = "logit"
            elif minv is not None and minv >= 0.0:
                link = "log1p"
            # override with rankgauss if requested (except probability columns)
            if prefer_rankgauss and not (minv is not None and 0.0 <= minv <= 1.0 and maxv <= 1.0):
                link = "rankgauss"
            continuous[c] = TypeSpec(kind="continuous", link=link, min_val=minv, max_val=maxv)
        else:
            categoricals.append(c)

    # apply transform overrides last (e.g., {'Attribute5': 'log1p'})
    if transform_overrides:
        for col, ln in transform_overrides.items():
            if col in continuous and ln in LINKS:
                continuous[col].link = ln

    return Schema(continuous=continuous, categoricals=categoricals)
