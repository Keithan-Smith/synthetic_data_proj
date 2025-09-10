import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

def membership_inference_auc(train_df: pd.DataFrame, synth_df: pd.DataFrame, cont_cols):
    real = train_df[cont_cols].copy(); real["is_real"] = 1
    syn  = synth_df[cont_cols].copy(); syn["is_real"] = 0
    data = pd.concat([real, syn], axis=0).dropna()
    y = data["is_real"].to_numpy(); X = data.drop(columns=["is_real"]).to_numpy()
    if len(set(y))<2: return float('nan')
    X, y = shuffle(X, y, random_state=7)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    scores = clf.predict_proba(X)[:,1]
    return float(roc_auc_score(y, scores))

def distance_to_closest_record(train_df: pd.DataFrame, synth_df: pd.DataFrame, cont_cols, seed: int = 0):
    """Standardized Euclidean DCR (synth→real) with a real→real baseline."""
    cols = [c for c in cont_cols if c in train_df.columns and c in synth_df.columns]
    if not cols:
        return {"dcr_error": "no common continuous columns"}

    Xr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    Xs = synth_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    mu = Xr.mean(axis=0, keepdims=True)
    sd = Xr.std(axis=0, keepdims=True); sd[sd==0.0] = 1.0
    Xr = (Xr - mu) / sd
    Xs = (Xs - mu) / sd

    # synth→real min distances
    d_sr = []
    for x in Xs:
        d = np.sqrt(((Xr - x) ** 2).sum(axis=1)).min()
        d_sr.append(d)
    d_sr = np.array(d_sr)

    # real→real baseline (sampled, LOO approx)
    n = min(len(Xr), 2000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xr), size=n, replace=False)
    Xb = Xr[idx]
    d_rr = []
    for i, x in enumerate(Xb):
        d = np.sqrt(((Xb - x) ** 2).sum(axis=1))
        d[i] = np.inf
        d_rr.append(d.min())
    d_rr = np.array(d_rr)

    out = {
        "dcr_synth_mean": float(np.mean(d_sr)),
        "dcr_synth_p1": float(np.quantile(d_sr, 0.01)),
        "dcr_synth_p5": float(np.quantile(d_sr, 0.05)),
        "dcr_synth_p10": float(np.quantile(d_sr, 0.10)),
        "dcr_real_mean": float(np.mean(d_rr)),
        "dcr_real_p1": float(np.quantile(d_rr, 0.01)),
        "dcr_real_p5": float(np.quantile(d_rr, 0.05)),
        "dcr_real_p10": float(np.quantile(d_rr, 0.10)),
    }
    tau = out["dcr_real_p1"]
    out["reident_rate_tau_p1"] = float(np.mean(d_sr < tau))
    return out

