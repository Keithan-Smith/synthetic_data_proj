"""
Downstream utility: train on synthetic, test on real (or vice-versa).
- Auto-detect binary target from a preferred list or first usable binary col
- Accept explicit `target_col` override
- Scales continuous, one-hots categoricals, aligns columns
- Higher max_iter to avoid convergence warnings
- Robust to column mismatches (adds missing test columns; drops unusable numerics)
- Excludes PD/target fields from automatic feature selection
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

# Columns never to use as features when auto-picking
_RESERVED = {
    "bad_within_horizon", "default_flag", "default", "class", "label", "target", "creditability",
    "pd_12m", "pd_monthly", "pd", "pd_score", "pd_12m_cal", "pd_monthly_cal"
}


# ---------------- helpers ----------------
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
    return s.astype(int)

def _pick_target(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        try:
            y = _binarize(df[explicit])
            if y.dropna().nunique() == 2:
                return explicit
        except Exception:
            pass
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

def _split_cols(df: pd.DataFrame, target_col: str,
                cont_cols: Optional[List[str]], cat_cols: Optional[List[str]]):
    """
    If feature lists are not provided, auto-derive them,
    excluding reserved PD/target-like fields.
    """
    if cont_cols is None:
        cont_cols = [c for c in df.columns
                     if c != target_col
                     and c not in _RESERVED
                     and not isinstance(df[c], pd.DataFrame)
                     and pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]
    if cat_cols is None:
        cat_cols  = [c for c in df.columns
                     if c != target_col
                     and c not in _RESERVED
                     and not isinstance(df[c], pd.DataFrame)
                     and not pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]
    return cont_cols, cat_cols

def _safe_get_series(df: pd.DataFrame, col: str, default_value):
    """Return df[col] if present; otherwise a Series filled with default_value."""
    if col in df.columns:
        return _ensure_series(df[col]).rename(col)
    return pd.Series(default_value, index=df.index, name=col)

def _build_design(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  cont_cols: List[str], cat_cols: List[str]):
    # --- Continuous block ---
    Xc_tr = pd.DataFrame(index=train_df.index)
    Xc_te = pd.DataFrame(index=test_df.index)
    if cont_cols:
        # use only columns present in train (we can fabricate test counterparts)
        cont_use = [c for c in cont_cols if c in train_df.columns]
        if cont_use:
            Xc_tr_raw = pd.concat([_safe_get_series(train_df, c, np.nan) for c in cont_use], axis=1)
            Xc_te_raw = pd.concat([_safe_get_series(test_df,  c, np.nan) for c in cont_use], axis=1)

            # to numeric + compute medians on train
            Xc_tr_raw = Xc_tr_raw.apply(pd.to_numeric, errors="coerce")
            Xc_te_raw = Xc_te_raw.apply(pd.to_numeric, errors="coerce")

            med = Xc_tr_raw.median(numeric_only=True)

            # drop columns whose median is NaN (uninformative numerics)
            good_cols = med.dropna().index.tolist()
            if good_cols:
                Xc_tr_raw = Xc_tr_raw[good_cols].fillna(med[good_cols])
                Xc_te_raw = Xc_te_raw[good_cols].fillna(med[good_cols])

                scaler = StandardScaler(with_mean=True, with_std=True)
                Xc_tr = pd.DataFrame(scaler.fit_transform(Xc_tr_raw), columns=good_cols, index=train_df.index)
                Xc_te = pd.DataFrame(scaler.transform(Xc_te_raw),   columns=good_cols, index=test_df.index)

    # --- Categorical block ---
    Xcat_tr = pd.DataFrame(index=train_df.index)
    Xcat_te = pd.DataFrame(index=test_df.index)
    if cat_cols:
        # use columns present in either frame; fabricate "__MISSING__" where absent
        cat_use = [c for c in cat_cols if (c in train_df.columns) or (c in test_df.columns)]
        if cat_use:
            tr_raw = pd.concat([_safe_get_series(train_df, c, "__MISSING__").astype(str) for c in cat_use], axis=1)
            te_raw = pd.concat([_safe_get_series(test_df,  c, "__MISSING__").astype(str) for c in cat_use], axis=1)

            Xcat_tr = pd.get_dummies(tr_raw, columns=cat_use, drop_first=False)
            Xcat_te = pd.get_dummies(te_raw, columns=cat_use, drop_first=False)

            # align test to train columns (drop extras in test; add zeros for missing)
            if Xcat_tr.shape[1]:
                Xcat_te = Xcat_te.reindex(columns=Xcat_tr.columns, fill_value=0)

    # Combine
    X_tr = pd.concat([Xc_tr, Xcat_tr], axis=1)
    X_te = pd.concat([Xc_te, Xcat_te], axis=1)
    return X_tr, X_te


# ---------------- main API ----------------
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
    Robust to column mismatches and excludes PD/targets when auto-selecting features.
    """
    # choose target (on test_df)
    tcol = _pick_target(test_df, target_col)
    if not tcol:
        return {"error": "No binary target found in test_df.", "auc": None, "brier": None, "acc": None}

    # binarize targets
    y_tr = _binarize(train_df[tcol]) if tcol in train_df.columns else None
    y_te = _binarize(test_df[tcol])

    # decide fit set (train_df if it has a usable target; otherwise test_df)
    if y_tr is None or y_tr.dropna().nunique() != 2:
        fit_df = test_df
        y_fit  = y_te
        score_df = test_df
        y_true   = y_te
    else:
        fit_df = train_df
        y_fit  = y_tr
        score_df = test_df
        y_true   = y_te

    # drop rows with missing targets
    fit_mask   = y_fit.notna()
    fit_df     = fit_df.loc[fit_mask]
    y_fit      = y_fit.loc[fit_mask]

    score_mask = y_true.notna()
    score_df   = score_df.loc[score_mask]
    y_true     = y_true.loc[score_mask]

    # feature lists (exclude reserved if auto)
    cont_cols, cat_cols = _split_cols(fit_df, tcol, cont_cols, cat_cols)

    # design matrices (robust to missing columns)
    X_tr, X_te = _build_design(fit_df, score_df, cont_cols, cat_cols)

    if X_tr.shape[1] == 0:
        return {
            "target_col": tcol, "auc": None, "brier": None, "acc": None,
            "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
            "n_features": 0, "error": "No usable features after preprocessing."
        }

    # LogisticRegression WITHOUT multi_class arg
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced",
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
        "used_cont_cols": [c for c in X_tr.columns if c in cont_cols],
        "used_cat_cols":  [c for c in X_tr.columns if c not in cont_cols],
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
