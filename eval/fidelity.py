import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

def univariate_fidelity(real: pd.DataFrame, synth: pd.DataFrame, cont_cols):
    rows = []
    for c in cont_cols:
        r = real[c].dropna().to_numpy() if c in real.columns else np.array([])
        s = synth[c].dropna().to_numpy() if c in synth.columns else np.array([])
        if len(r)>5 and len(s)>5:
            ks = ks_2samp(r, s).statistic
            wd = wasserstein_distance(r, s)
        else:
            ks, wd = np.nan, np.nan
        rows.append({"feature": c, "ks": ks, "wasserstein": wd,
                     "real_mean": np.mean(r) if len(r) else np.nan,
                     "synth_mean": np.mean(s) if len(s) else np.nan})
    return pd.DataFrame(rows)

def correlation_diff(real: pd.DataFrame, synth: pd.DataFrame, cols):
    R = real[cols].corr().to_numpy()
    S = synth[cols].corr().to_numpy()
    diff = np.nanmean(np.abs(R - S))
    return {"avg_abs_corr_diff": float(diff)}
