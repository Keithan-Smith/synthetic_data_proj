import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

def _find_binary_target(df: pd.DataFrame):
    preferred = ["bad_within_horizon","mortality_30d","readmission_30d","target","label","class","default","bad"]
    for p in preferred:
        if p in df.columns and df[p].dropna().nunique() == 2:
            return p
    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c
    return None

def generic_binary_downstream_eval(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tgt = _find_binary_target(test_df)
    if not tgt or tgt not in train_df.columns or tgt not in test_df.columns:
        return {"auc": float('nan'), "brier": float('nan')}
    feats = [c for c in train_df.columns if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c]) and c != tgt]
    if not feats:
        return {"auc": float('nan'), "brier": float('nan')}
    Xtr = train_df[feats].to_numpy(); ytr = train_df[tgt].to_numpy()
    Xte = test_df[feats].to_numpy();  yte = test_df[tgt].to_numpy()
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"auc": float('nan'), "brier": float('nan')}
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba); brier = brier_score_loss(yte, proba)
    return {"auc": float(auc), "brier": float(brier)}
