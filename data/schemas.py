from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd

@dataclass
class TypeSpec:
    kind: str
    link: str = ""
    min_val: float = None
    max_val: float = None

@dataclass
class Schema:
    continuous: Dict[str, TypeSpec] = field(default_factory=dict)
    categoricals: List[str] = field(default_factory=list)

    def cont_cols(self): return list(self.continuous.keys())
    def cat_cols(self):  return list(self.categoricals)

def log_link(x): return np.log(np.maximum(x, 1e-8))
def inv_log_link(x): return np.exp(x)
def logit_link(x):
    x = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(x/(1-x))
def inv_logit_link(x):
    y = 1/(1+np.exp(-x))
    return np.clip(y, 1e-8, 1-1e-8)

LINKS = {"": (lambda x: x, lambda x: x), "log": (log_link, inv_log_link), "logit": (logit_link, inv_logit_link)}

def forward_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c in out.columns:
            f,_ = LINKS.get(spec.link, LINKS[""]); out[c] = f(out[c].to_numpy())
    return out

def inverse_transform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c, spec in schema.continuous.items():
        if c in out.columns:
            _,inv = LINKS.get(spec.link, LINKS[""]); out[c] = inv(out[c].to_numpy())
            if spec.min_val is not None: out[c] = np.maximum(out[c], spec.min_val)
            if spec.max_val is not None: out[c] = np.minimum(out[c], spec.max_val)
    return out

def infer_schema_from_df(df: pd.DataFrame, max_categorical_card=50) -> Schema:
    continuous, categoricals = {}, []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            minv = float(np.nanmin(s.to_numpy())) if s.notna().any() else None
            maxv = float(np.nanmax(s.to_numpy())) if s.notna().any() else None
            link = ""
            if minv is not None and minv >= 0:
                try:
                    q10 = s.quantile(0.1); q90 = s.quantile(0.9)
                    if q90 > 0 and (q90 - q10) > 5 * max(1e-6, q10):
                        link = "log"
                except Exception: pass
            if maxv is not None and minv is not None and 0 <= minv <= 1 and maxv <= 1:
                link = "logit"
            continuous[c] = TypeSpec("continuous", link, minv, maxv)
        else:
            categoricals.append(c)
    return Schema(continuous=continuous, categoricals=categoricals)
