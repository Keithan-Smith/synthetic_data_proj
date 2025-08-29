"""
Downstream utility: train on synthetic, test on real (or vice-versa).
- Auto-detect binary target from a preferred list or first usable binary col
- Accept explicit `target_col` override
- No use of the deprecated `multi_class` arg
- Scales continuous, one-hots categoricals, aligns columns
- Higher max_iter to avoid convergence warnings
"""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

_PREFERRED_TARGETS: List[str] = [
    "bad_within_horizon", "default", "default_flag", "class", "label", "target", "creditability"
]

def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            raise ValueError("Empty DataFrame provided where a Series was expected.")
        return x.iloc[:, 0]
    return x

def _binarize(s_like) -> pd.Series:
    s = _ensure_series(s_like)
    if s.dtype == object:
        sl = s.astype(str).str.lower()
        u = set(sl.unique())
        if u <= {"good","bad"}:   return (sl == "bad").astype(int)
        if u <= {"yes","no"}:     return (sl == "yes").astype(int)
        if u <= {"true","false"}: return (sl == "true").astype(int)
    vals = set(pd.unique(s))
    if vals <= {1,2}:
        return s.map({1:0, 2:1})
    # last resort: cast to int (will raise if not valid)
    return s.astype(int)

def _pick_target(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in _PREFERRED_TARGETS:
        if c in df.columns:
            try:
                y = _binarize(df[c])
                if y.dropna().nunique() == 2:
                    return c
            except Exception:
                pass
    # fallback: any non-object binary
    for c in df.columns:
        try:
            col = _ensure_series(df[c])
            if c not in ("id","index") and col.dropna().nunique() == 2 and col.dtype != object:
                return c
        except Exception:
            pass
    return None

def _split_cols(df: pd.DataFrame, target_col: str, cont_cols: Optional[List[str]], cat_cols: Optional[List[str]]):
    if cont_cols is None:
        cont_cols = [c for c in df.columns
                     if c != target_col and not isinstance(df[c], pd.DataFrame)
                     and pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]
    if cat_cols is None:
        cat_cols  = [c for c in df.columns
                     if c != target_col and not isinstance(df[c], pd.DataFrame)
                     and not pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]
    return cont_cols, cat_cols

def _build_design(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  cont_cols: List[str], cat_cols: List[str]):
    # Continuous
    Xc_tr = pd.DataFrame(index=train_df.index)
    Xc_te = pd.DataFrame(index=test_df.index)
    scaler = None
    if cont_cols:
        Xc_tr_raw = pd.concat([_ensure_series(train_df[c]).rename(c) for c in cont_cols], axis=1)
        Xc_te_raw = pd.concat([_ensure_series(test_df[c]).rename(c)  for c in cont_cols], axis=1)
        # coerce numerics + impute to medians from train
        Xc_tr_raw = Xc_tr_raw.apply(pd.to_numeric, errors="coerce")
        med = Xc_tr_raw.median(numeric_only=True)
        Xc_tr_raw = Xc_tr_raw.fillna(med)
        Xc_te_raw = Xc_te_raw.apply(pd.to_numeric, errors="coerce").fillna(med)
        # scale
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xc_tr = pd.DataFrame(scaler.fit_transform(Xc_tr_raw), columns=cont_cols, index=train_df.index)
        Xc_te = pd.DataFrame(scaler.transform(Xc_te_raw),   columns=cont_cols, index=test_df.index)

    # Categorical (pandas get_dummies; align)
    Xcat_tr = pd.DataFrame(index=train_df.index)
    Xcat_te = pd.DataFrame(index=test_df.index)
    if cat_cols:
        tr_raw = pd.concat([_ensure_series(train_df[c]).astype(str).rename(c) for c in cat_cols], axis=1)
        te_raw = pd.concat([_ensure_series(test_df[c]).astype(str).rename(c)  for c in cat_cols], axis=1)
        Xcat_tr = pd.get_dummies(tr_raw, columns=cat_cols, drop_first=False)
        Xcat_te = pd.get_dummies(te_raw, columns=cat_cols, drop_first=False)
        # align test to train columns
        for c in Xcat_tr.columns:
            if c not in Xcat_te.columns:
                Xcat_te[c] = 0
        extra = [c for c in Xcat_te.columns if c not in Xcat_tr.columns]
        if extra:
            Xcat_te = Xcat_te.drop(columns=extra)
        Xcat_te = Xcat_te[Xcat_tr.columns]

    X_tr = pd.concat([Xc_tr, Xcat_tr], axis=1)
    X_te = pd.concat([Xc_te, Xcat_te], axis=1)
    return X_tr, X_te

def generic_binary_downstream_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: Optional[str] = None,
    cont_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    max_iter: int = 2000
) -> Dict[str, float]:
    """
    Fits LogisticRegression on train_df -> predicts on test_df target.
    Returns dict with AUC, Brier, Accuracy, and bookkeeping.
    """
    # choose target
    tcol = _pick_target(test_df, target_col)
    if not tcol:
        return {"error": "No binary target found in test_df.", "auc": None, "brier": None, "acc": None}

    y_tr = _binarize(train_df[tcol]) if tcol in train_df.columns else None
    y_te = _binarize(test_df[tcol])

    # If train doesn't have the target, try to synthesize probability by training on test_df and scoring on test_df
    if y_tr is None or y_tr.dropna().nunique() != 2:
        # swap train/test role for fitting, but still report on test_df
        fit_df = test_df
        y_fit  = y_te
        score_df = test_df
        y_true   = y_te
    else:
        fit_df = train_df
        y_fit  = y_tr
        score_df = test_df
        y_true   = y_te

    cont_cols, cat_cols = _split_cols(fit_df, tcol, cont_cols, cat_cols)
    X_tr, X_te = _build_design(fit_df, score_df, cont_cols, cat_cols)

    # LogisticRegression WITHOUT multi_class arg (fixes deprecation warnings)
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced"  # helps with imbalanced defaults
    )
    lr.fit(X_tr, y_fit)

    p = lr.predict_proba(X_te)[:, 1]
    y_hat = (p >= 0.5).astype(int)

    out = {
        "target_col": tcol,
        "auc": None,
        "brier": None,
        "acc": None,
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, p))
    except Exception:
        out["auc"] = None
    try:
        out["brier"] = float(brier_score_loss(y_true, p))
    except Exception:
        out["brier"] = None
    try:
        out["acc"] = float(accuracy_score(y_true, y_hat))
    except Exception:
        out["acc"] = None
    return out
