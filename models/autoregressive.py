import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model")

class CategoricalAR:
    """
    Autoregressive categorical sampler:
      - fit(): for each cat col in order, train multinomial LR on [cont + prev cats (one-hot)]
      - sample(n, cond_cont, logit_bias): sequentially sample cat cols; optional probability tilts
    logit_bias format: {col: {category_value: delta_logit, ...}}
    """
    def __init__(self, cat_cols, cont_cols):
        self.cat_cols = list(cat_cols)
        self.cont_cols = list(cont_cols)
        # For each categorical column: (model, feat_cols_used, scaler_for_cont, classes_in_order, feature_names)
        self.models = {}

    def fit(self, df: pd.DataFrame):
        # if no cat cols, nothing to train
        if not self.cat_cols:
            self.models = {}
            return self

        df = df[self.cont_cols + self.cat_cols].dropna().reset_index(drop=True)

        # One-hot for all cats once so we know possible previous dummies
        if self.cat_cols:
            dummies_all = pd.get_dummies(df[self.cat_cols], columns=self.cat_cols, drop_first=False)
        else:
            dummies_all = pd.DataFrame(index=df.index)

        seen = []
        for col in self.cat_cols:
            # features = continuous + one-hots of previously seen cats
            if seen and not dummies_all.empty:
                feat_cols = [c for c in dummies_all.columns if any(c.startswith(s + "_") for s in seen)]
                X_prev = dummies_all[feat_cols] if feat_cols else pd.DataFrame(index=df.index)
            else:
                feat_cols = []
                X_prev = pd.DataFrame(index=df.index)

            X_cont = df[self.cont_cols] if self.cont_cols else pd.DataFrame(index=df.index)
            X = pd.concat([X_cont, X_prev], axis=1)

            scaler = None
            if not X_cont.empty:
                scaler = StandardScaler(with_mean=True, with_std=True)
                X_scaled = X.copy()
                X_scaled[self.cont_cols] = scaler.fit_transform(X_cont)
                X = X_scaled

            y = df[col].astype(str)
            # In rare cases a column might be constant; fall back to prior-only
            if y.nunique() <= 1:
                prior = {y.iloc[0]: 1.0}
                self.models[col] = {
                    "lr": None, "feat_cols": feat_cols, "scaler": scaler,
                    "classes": list(prior.keys()), "feature_names": list(X.columns),
                    "prior": prior
                }
            else:
                lr = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto")
                lr.fit(X, y)
                self.models[col] = {
                    "lr": lr, "feat_cols": feat_cols, "scaler": scaler,
                    "classes": lr.classes_.tolist(), "feature_names": list(X.columns),
                    "prior": None
                }
            seen.append(col)
        return self

    def _tilt_probs(self, classes, probs, bias_map):
        if not bias_map:
            return probs
        w = []
        for cls, p in zip(classes, probs):
            delta = float(bias_map.get(cls, 0.0))
            w.append(p * np.exp(delta))
        w = np.asarray(w, dtype=float)
        s = w.sum()
        return (w / s) if s > 0 else np.ones_like(w) / len(w)

    def sample(self, n: int, cond_cont: pd.DataFrame, logit_bias=None, **_):
        logit_bias = logit_bias or {}

        # no categorical columns → return empty DF with n rows
        if not self.cat_cols or not self.models:
            return pd.DataFrame(index=range(n))

        out_rows = []
        for i in range(n):
            row_prev = {}  # sampled cats so far for this row

            for col in self.cat_cols:
                m = self.models[col]
                lr         = m["lr"]
                feat_cols  = m["feat_cols"]
                scaler     = m["scaler"]
                classes    = m["classes"]
                feat_names = m["feature_names"]
                prior      = m.get("prior")

                # one-hot for previously sampled cats on this row
                if row_prev:
                    df_prev = pd.DataFrame([{k: row_prev[k] for k in row_prev.keys()}])
                    # only call get_dummies if there are columns to encode
                    if df_prev.shape[1] > 0:
                        prev_oh = pd.get_dummies(df_prev, columns=df_prev.columns, drop_first=False)
                    else:
                        prev_oh = pd.DataFrame([[]])
                else:
                    prev_oh = pd.DataFrame([[]])

                # align to training one-hot columns for prev cats
                if feat_cols:
                    for c in feat_cols:
                        if c not in prev_oh.columns:
                            prev_oh[c] = 0
                    prev_oh = prev_oh[feat_cols]
                else:
                    prev_oh = pd.DataFrame([[]])

                # continuous features for this row
                x_cont = cond_cont.iloc[[i]][self.cont_cols] if self.cont_cols else pd.DataFrame([[]])
                if scaler is not None and not x_cont.empty:
                    x_cont[self.cont_cols] = scaler.transform(x_cont[self.cont_cols])

                X = pd.concat([x_cont.reset_index(drop=True), prev_oh.reset_index(drop=True)], axis=1)

                # ensure exact training column order (missing → 0)
                for c in feat_names:
                    if c not in X.columns:
                        X[c] = 0
                X = X[feat_names] if feat_names else pd.DataFrame([[]])

                # predict class probabilities (or use prior)
                if prior is not None:
                    classes_arr = np.array(classes)
                    probs = np.array([prior.get(cls, 0.0) for cls in classes_arr], dtype=float)
                    s = probs.sum()
                    probs = probs / s if s > 0 else np.ones_like(probs)/len(probs)
                else:
                    probs = lr.predict_proba(X)[0]

                # apply optional bias
                probs = self._tilt_probs(classes, probs, logit_bias.get(col, {}))

                choice = np.random.choice(classes, p=probs)
                row_prev[col] = choice

            out_rows.append(row_prev)

        return pd.DataFrame(out_rows, index=range(n))
