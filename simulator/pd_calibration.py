"""
Probability-of-Default (PD) calibration for a tabular dataset.

- Preserves robust behavior from your previous version:
  * Auto target pick (or pass target_col explicitly)
  * Binarization of {good/bad, yes/no, true/false, 1/2}
  * Continuous means cached for imputing at predict time
  * Categorical 'UNK' fallback + one-hot alignment to training levels
  * Feature column order preserved; missing cols filled with 0
  * Base-rate fallback if model unavailable

- Fixes/Improvements:
  * No 'multi_class' arg (removes sklearn deprecation warnings)
  * Standardize numerics with StandardScaler
  * Higher max_iter for better convergence
  * Class imbalance handled with class_weight="balanced"
  * Returns monthly hazard h from PD_T via h = -ln(1 - PD_T)/T

Usage:
    pdcal = PDCalibrator(cont_cols=[...], cat_cols=[...], horizon_months=12)
    pdcal.fit(df, target_col="bad_within_horizon")
    pd_m = pdcal.monthly_hazard(book_df)
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

_PREFERRED_TARGETS = [
    "bad_within_horizon","default","default_flag","class","label","target"
]

def _pick_target(df: pd.DataFrame) -> Optional[str]:
    for c in _PREFERRED_TARGETS:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                continue
            if c not in ("id","index") and col.dropna().nunique() == 2 and col.dtype != object:
                return c
        except Exception:
            pass
    return None

def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            raise ValueError("Empty DataFrame passed for target column.")
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
    if vals <= {1,2}: return s.map({1:0, 2:1})
    return s.astype(int)

class PDCalibrator:
    """
    Fit PD over T months (default 12) and expose monthly hazard via:
      PD_T = 1 - exp(-h T)  =>  h = -ln(1 - PD_T)/T

    Convergence fixes:
      - Standardize continuous
      - Try lbfgs, fallback to saga if needed
      - No deprecated `multi_class` arg
    """
    def __init__(self, cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 horizon_months: int = 12):
        self.cont_cols = list(cont_cols) if cont_cols else []
        self.cat_cols  = list(cat_cols)  if cat_cols  else []
        self.horizon_months = int(horizon_months)

        self.scaler: Optional[StandardScaler] = None
        self.model:  Optional[LogisticRegression] = None
        self.cat_levels: List[str] = []
        self.feature_cols: List[str] = []
        self.base_rate: Optional[float] = None
        self.target_col_: Optional[str] = None
        self.cont_means_: dict = {}

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        tcol = target_col or _pick_target(df)
        if not tcol:
            raise ValueError("PDCalibrator.fit: could not find a binary target column.")
        self.target_col_ = tcol

        y = _binarize(df[tcol]).astype(int)

        if not self.cont_cols:
            self.cont_cols = [c for c in df.columns
                              if c != tcol and not isinstance(df[c], pd.DataFrame)
                              and pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]
        if not self.cat_cols:
            self.cat_cols  = [c for c in df.columns
                              if c != tcol and not isinstance(df[c], pd.DataFrame)
                              and not pd.api.types.is_numeric_dtype(_ensure_series(df[c]))]

        # Continuous (+ means for imputation)
        X_cont = pd.DataFrame(index=df.index)
        if self.cont_cols:
            cont_series = []
            for c in self.cont_cols:
                s = _ensure_series(df[c])
                cont_series.append(s.rename(c))
            X_cont = pd.concat(cont_series, axis=1)
            self.cont_means_ = {c: float(pd.to_numeric(X_cont[c], errors="coerce").mean()) for c in self.cont_cols}
        else:
            self.cont_means_ = {}

        # Categorical (alignable OHE)
        X_cat = pd.DataFrame(index=df.index)
        if self.cat_cols:
            cat_series = []
            for c in self.cat_cols:
                s = _ensure_series(df[c]).astype(str)
                cat_series.append(s.rename(c))
            X_cat_raw = pd.concat(cat_series, axis=1)
            X_cat = pd.get_dummies(X_cat_raw, columns=self.cat_cols, drop_first=True)  # drop_first=True to reduce collinearity
            self.cat_levels = list(X_cat.columns)
        else:
            self.cat_levels = []

        # Scale continuous
        if not X_cont.empty:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            X_cont = pd.DataFrame(self.scaler.fit_transform(
                                  pd.DataFrame({c: pd.to_numeric(X_cont[c], errors="coerce").fillna(self.cont_means_[c])
                                                for c in self.cont_cols})),
                                  columns=self.cont_cols, index=df.index)
        else:
            self.scaler = None

        X = pd.concat([X_cont, X_cat], axis=1)
        self.feature_cols = list(X.columns)

        # Try lbfgs → saga fallback if needed
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", category=ConvergenceWarning)
            self.model = LogisticRegression(solver="lbfgs", penalty="l2",
                                            class_weight="balanced", max_iter=2000)
            self.model.fit(X, y)
            conv_warn = any(isinstance(w.message, ConvergenceWarning) for w in rec)

        if conv_warn:
            self.model = LogisticRegression(solver="saga", penalty="l2",
                                            class_weight="balanced", C=0.5, max_iter=8000)
            self.model.fit(X, y)

        self.base_rate = float(np.clip(y.mean(), 1e-6, 1-1e-6))
        return self

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        # Continuous with safe fallback + scaling
        X_cont = pd.DataFrame(index=df.index)
        if self.cont_cols:
            cont_series = []
            n = len(df)
            for c in self.cont_cols:
                if c in df.columns:
                    s = _ensure_series(df[c])
                else:
                    mean = self.cont_means_.get(c, 0.0)
                    s = pd.Series([mean]*n, index=df.index, name=c)
                cont_series.append(s.rename(c))
            X_cont_raw = pd.concat(cont_series, axis=1)
            X_cont_raw = pd.DataFrame({c: pd.to_numeric(X_cont_raw[c], errors="coerce").fillna(self.cont_means_[c])
                                       for c in self.cont_cols}, index=df.index)
            if self.scaler is not None:
                X_cont = pd.DataFrame(self.scaler.transform(X_cont_raw),
                                      columns=self.cont_cols, index=df.index)
            else:
                X_cont = X_cont_raw

        # Categoricals with 'UNK' fallback, and align to training cols
        X_cat = pd.DataFrame(index=df.index)
        if self.cat_cols:
            cat_series = []
            n = len(df)
            for c in self.cat_cols:
                if c in df.columns:
                    s = _ensure_series(df[c]).astype(str)
                else:
                    s = pd.Series(["UNK"]*n, index=df.index, name=c)
                cat_series.append(s.rename(c))
            X_cat_raw = pd.concat(cat_series, axis=1)
            X_cat = pd.get_dummies(X_cat_raw, columns=self.cat_cols, drop_first=True)
            for c in self.cat_levels:
                if c not in X_cat.columns:
                    X_cat[c] = 0
            X_cat = X_cat[self.cat_levels] if self.cat_levels else X_cat

        X = pd.concat([X_cont, X_cat], axis=1)
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[self.feature_cols]
        return X

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            base = self.base_rate if self.base_rate is not None else 0.05
            return np.full(len(df), base, dtype=float)
        X = self._design(df)
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(p, 1e-6, 1-1e-6)

    def monthly_hazard(self, df: pd.DataFrame) -> np.ndarray:
        pT = self.predict_pd(df)
        T = max(1, self.horizon_months)
        h = -np.log(1.0 - pT) / T
        return np.clip(h, 1e-6, 0.95)

    def expected_columns(self) -> List[str]:
        return list(self.cont_cols) + list(self.cat_cols)
