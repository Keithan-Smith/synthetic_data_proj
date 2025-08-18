import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

class CategoricalAR:
    """Autoregressive model over categorical columns given continuous block.
    Trains one multinomial logistic model per categorical in a fixed order.
    """
    def __init__(self, cat_cols, cont_cols):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.encoders = {c: OneHotEncoder(handle_unknown='ignore', sparse_output=False) for c in cat_cols}
        self.models = {}

    def _features(self, df, upto_index):
        # Build features: continuous + one-hots of previous categoricals
        Xc = df[self.cont_cols].to_numpy()
        prev_oh = []
        for c in self.cat_cols[:upto_index]:
            oh = self.encoders[c].transform(df[[c]])
            prev_oh.append(oh)
        X = np.hstack([Xc] + prev_oh) if prev_oh else Xc
        return X 

    def fit(self, df: pd.DataFrame):
        for i, c in enumerate(self.cat_cols):
            # fit encoder for current target
            self.encoders[c].fit(df[[c]])
            X = self._features(df, i)
            y = df[c].astype(str).to_numpy()
            clf = LogisticRegression(max_iter=200, multi_class='multinomial')
            clf.fit(X, y)
            self.models[c] = clf
        return self

    def sample(self, n: int, cont_df: pd.DataFrame) -> pd.DataFrame:
        # start with continuous block replicated n times
        out = cont_df[self.cont_cols].reset_index(drop=True).copy()
        for i, c in enumerate(self.cat_cols):
            X = self._features(out, i)
            probs = self.models[c].predict_proba(X)
            classes = self.models[c].classes_
            draws = [np.random.choice(classes, p=p) for p in probs]
            out[c] = draws
        return out[self.cat_cols]