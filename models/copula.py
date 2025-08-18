import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cholesky

class GaussianCopula:
    def __init__(self, cols):
        self.cols = cols
        self.mu = None
        self.corr = None
        self.quantiles_ = {}
        self.eps = 1e-6

    def _to_uniform(self, x: np.ndarray) -> np.ndarray:
        # empirical CDF per column
        u = np.zeros_like(x, dtype=float)
        for j in range(x.shape[1]):
            col = x[:, j]
            ranks = pd.Series(col).rank(method='average').to_numpy()
            u[:, j] = (ranks - 0.5) / len(col)
            u[:, j] = np.clip(u[:, j], self.eps, 1 - self.eps)
        return u

    def fit(self, df: pd.DataFrame):
        X = df[self.cols].to_numpy().astype(float)
        # store quantiles for inverse mapping
        for c in self.cols:
            self.quantiles_[c] = np.sort(df[c].to_numpy())
        U = self._to_uniform(X)
        Z = norm.ppf(U)
        self.mu = Z.mean(axis=0)
        Zc = Z - self.mu
        self.corr = np.corrcoef(Zc, rowvar=False)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        L = cholesky(self.corr, lower=True)
        z = np.random.normal(size=(n, len(self.cols)))
        Z = (z @ L.T) + self.mu
        U = norm.cdf(Z)
        out = {}
        for j, c in enumerate(self.cols):
            quant = self.quantiles_[c]
            idx = np.clip((U[:, j] * (len(quant) - 1)).astype(int), 0, len(quant)-1)
            out[c] = quant[idx]
        return pd.DataFrame(out)
