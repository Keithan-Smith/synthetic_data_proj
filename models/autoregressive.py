from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


class _AlwaysClass:
    """Fallback when a categorical column has a single class in training."""
    def __init__(self, klass: str):
        self._klass = klass
        self.classes_ = np.array([klass], dtype=object)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._klass, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            raise ValueError("Empty DataFrame where a Series was expected.")
        return x.iloc[:, 0]
    return x


def _safe_to_str(s: pd.Series) -> pd.Series:
    return _ensure_series(s).astype(str).fillna("UNK")


class CategoricalAR:
    """
    Autoregressive model for categorical columns:
      y_i ~ [scaled continuous features] + OHE(previous categorical 0..i-1)

    Convergence & dtype handling:
      * Drop one OHE level per previous categorical (avoid collinearity)
      * Standardize continuous features with StandardScaler (fit & sample paths)
      * If lbfgs hits iteration limit, refit with saga + stronger regularization
      * **Dtype fix**: never assign scaler output into an int64 block in-place
    """

    def __init__(self, cat_cols: List[str], cont_cols: List[str]):
        self.cat_cols = list(cat_cols)
        self.cont_cols = list(cont_cols)

        self.fit_cat_cols: List[str] = list(cat_cols)
        self.fit_cont_cols: List[str] = list(cont_cols)

        self.models: Dict[str, object] = {}
        self.model_feature_cols: Dict[str, List[str]] = {}
        self.prev_levels: Dict[str, Dict[str, List[str]]] = {}  # full level list per prev-cat
        self.prev_drop_level: Dict[str, Dict[str, str]] = {}    # which level we drop for each prev-cat
        self.cont_means_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None

    # ---------- helpers ----------

    def _fit_cont_baseline(self, df_cont: Optional[pd.DataFrame]) -> None:
        means = {}
        for c in self.fit_cont_cols:
            if df_cont is not None and c in df_cont.columns:
                s = pd.to_numeric(_ensure_series(df_cont[c]), errors="coerce")
                means[c] = float(s.mean())
            else:
                means[c] = 0.0
        self.cont_means_ = means

        # Fit scaler on continuous block (if any)
        if self.fit_cont_cols:
            if df_cont is None or df_cont.empty:
                Xc = pd.DataFrame({c: np.full(1, self.cont_means_[c], dtype=np.float64)
                                   for c in self.fit_cont_cols})
            else:
                Xc = pd.DataFrame(
                    {
                        c: pd.to_numeric(_ensure_series(df_cont[c]), errors="coerce")
                           .fillna(self.cont_means_[c]).astype(np.float64).values
                        for c in self.fit_cont_cols
                    },
                    index=df_cont.index,
                )
            self.scaler_ = StandardScaler(with_mean=True, with_std=True).fit(Xc)
        else:
            self.scaler_ = None

    def _cont_design(self, df_cont: Optional[pd.DataFrame], index_like) -> pd.DataFrame:
        n = len(index_like)
        cols = {}
        for c in self.fit_cont_cols:
            if df_cont is not None and c in df_cont.columns:
                s = pd.to_numeric(_ensure_series(df_cont[c]), errors="coerce").fillna(self.cont_means_.get(c, 0.0))
                cols[c] = s.values
            else:
                cols[c] = np.full(n, self.cont_means_.get(c, 0.0))
        Xc = pd.DataFrame(cols, index=index_like)

        if self.scaler_ is not None and not Xc.empty:
            # dtype-safe: cast to float; preserve feature names by passing a DataFrame
            Xc = Xc.astype(np.float64, copy=False)
            Xc_scaled = self.scaler_.transform(Xc)          # <- NO .values here
            Xc = pd.DataFrame(Xc_scaled, columns=Xc.columns, index=index_like)
        return Xc


    @staticmethod
    def _ohe_frame_drop_first(df_cat_subset: pd.DataFrame,
                              per_col_levels: Dict[str, List[str]],
                              per_col_drop: Dict[str, str]) -> pd.DataFrame:
        """
        One-hot encode with a *stable dropped level* per categorical to avoid collinearity.
        """
        if df_cat_subset is None or df_cat_subset.shape[1] == 0:
            return pd.DataFrame(index=(df_cat_subset.index if df_cat_subset is not None else None))

        tmp = df_cat_subset.astype(str)
        cur = pd.get_dummies(tmp, columns=list(tmp.columns), drop_first=False)

        expected_cols: List[str] = []
        # Build expected columns by skipping the dropped level for each prev-cat
        for col, levels in per_col_levels.items():
            drop_lv = per_col_drop.get(col)
            for lv in levels:
                if lv == drop_lv:
                    continue  # drop one level
                expected_cols.append(f"{col}_{lv}")

        # Add missing expected columns
        for c in expected_cols:
            if c not in cur.columns:
                cur[c] = 0

        # Remove extras (including the dropped columns)
        extra = [c for c in cur.columns if c not in expected_cols]
        if extra:
            cur = cur.drop(columns=extra)
        cur = cur[expected_cols] if expected_cols else cur
        return cur

    def _try_fit_logreg(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
        """
        Try lbfgs then, on ConvergenceWarning, retry with saga + stronger regularization.
        """
        # First attempt: lbfgs
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", category=ConvergenceWarning)
            lr = LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=4000, multi_class="multinomial", class_weight="balanced")
            lr.fit(X, y)
            conv_warn = any(isinstance(w.message, ConvergenceWarning) for w in rec)

        if not conv_warn:
            return lr

        # Fallback: saga (handles large sparse-ish OHEs better), more iters, stronger reg
        lr2 = LogisticRegression(
        solver="saga", penalty="l2", C=0.5, max_iter=8000,
            multi_class="multinomial", class_weight="balanced"
            )
        lr2.fit(X, y)
        return lr2

    # ---------- API ----------

    def fit(self,
            df_cat_or_full: pd.DataFrame,
            df_cont: Optional[pd.DataFrame] = None) -> "CategoricalAR":
        """
        Backward compatible:
          - fit(df_full)           # old call sites
          - fit(df_cat, df_cont)   # new call sites
        """
        if not isinstance(df_cat_or_full, pd.DataFrame) or df_cat_or_full.empty:
            raise ValueError("CategoricalAR.fit: first argument must be a non-empty DataFrame.")

        # Split if df_cont not provided
        if df_cont is None:
            full = df_cat_or_full
            cat_present = [c for c in self.cat_cols if c in full.columns]
            cont_present = [c for c in self.cont_cols if c in full.columns]

            if len(cat_present) < len(self.cat_cols):
                missing = [c for c in self.cat_cols if c not in full.columns]
                if missing:
                    warnings.warn(f"CategoricalAR.fit: missing categorical columns {missing}; "
                                  f"fitting with {cat_present} only.")
            if not cat_present:
                raise ValueError("CategoricalAR.fit: no categorical columns from the configured "
                                 f"set were found in the provided DataFrame: expected {self.cat_cols}.")

            df_cat = full[cat_present].copy()
            df_cont = full[cont_present].copy() if cont_present else pd.DataFrame(index=full.index)

            self.fit_cat_cols = cat_present
            self.fit_cont_cols = cont_present
        else:
            df_cat = df_cat_or_full
            if not set(self.cat_cols).issubset(set(df_cat.columns)):
                cat_present = [c for c in self.cat_cols if c in df_cat.columns]
                missing = [c for c in self.cat_cols if c not in df_cat.columns]
                if missing:
                    warnings.warn(f"CategoricalAR.fit: missing categorical columns {missing}; "
                                  f"fitting with {cat_present} only.")
                if not cat_present:
                    raise ValueError("CategoricalAR.fit: no usable categorical columns were provided.")
                df_cat = df_cat[cat_present].copy()
                self.fit_cat_cols = cat_present
            else:
                self.fit_cat_cols = list(self.cat_cols)

            if df_cont is None:
                df_cont = pd.DataFrame(index=df_cat.index)
                self.fit_cont_cols = []
            else:
                cont_present = [c for c in self.cont_cols if c in df_cont.columns]
                if len(cont_present) < len(self.cont_cols):
                    missing = [c for c in self.cont_cols if c not in (df_cont.columns)]
                    if missing:
                        warnings.warn(f"CategoricalAR.fit: missing continuous columns {missing}; "
                                      f"fitting with {cont_present} only.")
                self.fit_cont_cols = cont_present

        # Prepare continuous baseline + scaler
        self._fit_cont_baseline(df_cont)

        # Fit per-categorical in order
        self.models.clear()
        self.model_feature_cols.clear()
        self.prev_levels.clear()
        self.prev_drop_level.clear()

        for i, tgt in enumerate(self.fit_cat_cols):
            y = _safe_to_str(df_cat[tgt])
            mask = y.notna()
            y_fit = y.loc[mask]

            prev_cols = self.fit_cat_cols[:i]
            prev_raw = df_cat[prev_cols].loc[mask] if prev_cols else pd.DataFrame(index=y_fit.index)

            # Levels for previous categoricals; record a dropped level to avoid collinearity
            levels_map: Dict[str, List[str]] = {}
            drop_map: Dict[str, str] = {}
            for pc in prev_cols:
                lv = _safe_to_str(prev_raw[pc]).unique().tolist()
                lv_sorted = sorted(map(str, lv))
                levels_map[pc] = lv_sorted
                drop_map[pc] = lv_sorted[0] if lv_sorted else "UNK"  # stable dropped level

            self.prev_levels[tgt] = levels_map
            self.prev_drop_level[tgt] = drop_map

            prev_ohe = self._ohe_frame_drop_first(prev_raw, levels_map, drop_map)
            X_cont = self._cont_design(df_cont.loc[mask] if df_cont is not None else None, y_fit.index)
            X_fit = pd.concat([X_cont, prev_ohe], axis=1)

            uniq = pd.unique(y_fit)
            if len(uniq) <= 1:
                self.models[tgt] = _AlwaysClass(str(uniq[0]))
                self.model_feature_cols[tgt] = list(X_fit.columns)
                continue

            lr = self._try_fit_logreg(X_fit, y_fit)
            self.models[tgt] = lr
            self.model_feature_cols[tgt] = list(X_fit.columns)

        return self

    def sample(self, n: int, cont_syn: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # output index
        if cont_syn is not None and isinstance(cont_syn, pd.DataFrame) and len(cont_syn) == n:
            index_like = cont_syn.index
        else:
            index_like = pd.RangeIndex(n)

        out = pd.DataFrame(index=index_like)
        X_cont_all = self._cont_design(cont_syn if cont_syn is not None else None, index_like)

        rng = np.random.default_rng()

        for i, tgt in enumerate(self.fit_cat_cols):
            model = self.models.get(tgt)
            if model is None:
                out[tgt] = "UNK"
                continue

            prev_cols = self.fit_cat_cols[:i]
            if prev_cols:
                prev_now = out[prev_cols].astype(str)
                prev_ohe = self._ohe_frame_drop_first(
                    prev_now,
                    self.prev_levels.get(tgt, {}),
                    self.prev_drop_level.get(tgt, {}),
                )
            else:
                prev_ohe = pd.DataFrame(index=index_like)

            X_design = pd.concat([X_cont_all, prev_ohe], axis=1)

            expected = self.model_feature_cols.get(tgt, list(X_design.columns))
            for cexp in expected:
                if cexp not in X_design.columns:
                    X_design[cexp] = 0
            extra = [c for c in X_design.columns if c not in expected]
            if extra:
                X_design = X_design.drop(columns=extra)
            X_design = X_design[expected]

            proba = model.predict_proba(X_design)
            classes = getattr(model, "classes_", None)

            if classes is None or proba.shape[1] != len(classes):
                out[tgt] = model.predict(X_design)
                continue

            p = np.asarray(proba, dtype=float)
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            draws = [rng.choice(classes, p=p[i]) for i in range(p.shape[0])]
            out[tgt] = np.array(draws, dtype=object)

        return out
