import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_PREFERRED = ["bad_within_horizon","default","default_flag","class","label","target"]

def _binarize(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        sl = s.astype(str).str.lower()
        if set(sl.unique()) <= {"good","bad"}:   return (sl=="bad").astype(int)
        if set(sl.unique()) <= {"yes","no"}:     return (sl=="yes").astype(int)
        if set(sl.unique()) <= {"true","false"}: return (sl=="true").astype(int)
    vals = set(pd.unique(s))
    if vals <= {1,2}: return s.map({1:0,2:1})
    return s.astype(int)

def _pick_target(df: pd.DataFrame) -> str:
    for c in _PREFERRED:
        if c in df.columns: return c
    # fallback: any binary column
    for c in df.columns:
        if df[c].dropna().nunique()==2: return c
    raise ValueError("No binary target column found.")

def generic_binary_downstream_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = None):
    # target selection
    if target_col is None:
        target_col = _pick_target(train_df)
        # ensure also exists in test_df
        if target_col not in test_df.columns:
            # try test_df pick; then align
            t2 = _pick_target(test_df)
            if t2 != target_col:
                # align by creating the chosen name on whichever is missing
                if target_col not in test_df.columns:
                    test_df = test_df.copy()
                    test_df[target_col] = _binarize(test_df[t2])

    # split features/labels
    y_tr = _binarize(train_df[target_col])
    y_te = _binarize(test_df[target_col])

    X_tr = train_df.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number])
    X_te = test_df.drop(columns=[target_col],  errors='ignore').select_dtypes(include=[np.number])

    # align numeric columns
    common = [c for c in X_tr.columns if c in X_te.columns]
    if not common:
        return {"auc": None, "brier": None, "note":"No overlapping numeric features."}
    X_tr = X_tr[common].fillna(X_tr.median(numeric_only=True))
    X_te = X_te[common].fillna(X_tr.median(numeric_only=True))

    # scale + LR
    sc = StandardScaler()
    X_trs = sc.fit_transform(X_tr)
    X_tes = sc.transform(X_te)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
    lr.fit(X_trs, y_tr)
    p = lr.predict_proba(X_tes)[:,1]

    return {
        "auc":  float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p))
    }
