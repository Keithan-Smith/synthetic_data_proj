import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .spec import ShockSpec

def build_corr_override(base_corr: np.ndarray,
                        cols: list,
                        spec: ShockSpec) -> np.ndarray:
    """Start from learned correlation -> shrink towards I -> apply pair overrides."""
    C = base_corr.copy()

    # shrink towards identity
    if spec.corr_shrink and spec.corr_shrink > 0.0:
        lam = float(np.clip(spec.corr_shrink, 0.0, 1.0))
        I = np.eye(C.shape[0])
        C = (1.0 - lam) * C + lam * I

    # targeted pair overrides
    for (a, b), rho in (spec.corr_pairs or {}).items():
        if a in cols and b in cols:
            i, j = cols.index(a), cols.index(b)
            C[i, j] = C[j, i] = float(rho)

    # sanitize
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C

def apply_cont_marginal_shocks(df_cont: pd.DataFrame, spec: ShockSpec) -> pd.DataFrame:
    out = df_cont.copy()
    for k, v in (spec.cont_mu_shift or {}).items():
        if k in out.columns:
            out[k] = out[k] + float(v)
    for k, s in (spec.cont_scale or {}).items():
        if k in out.columns:
            out[k] = out[k] * float(s)
    return out
