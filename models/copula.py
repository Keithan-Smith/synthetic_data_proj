import numpy as np
import pandas as pd
from typing import Optional, Sequence

class GaussianCopula:
    def __init__(self, cols: Optional[Sequence[str]] = None, jitter: float = 1e-6, random_state: int = 42):
        self.cols = list(cols) if cols is not None else None
        self.mean = None
        self.cov  = None
        self.corr = None
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(int(random_state))

    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        invD = np.diag(1.0 / std)
        C = invD @ cov @ invD
        C = (C + C.T) / 2.0
        np.fill_diagonal(C, 1.0)
        return C

    def fit(self, X_df: pd.DataFrame):
        if self.cols is None:
            self.cols = list(X_df.columns)
        X = X_df[self.cols].to_numpy(dtype=float, copy=True)
        self.mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = (cov + cov.T) / 2.0
        eig = np.linalg.eigvalsh(cov)
        if np.any(eig <= 0):
            cov = cov + np.eye(cov.shape[0]) * (self.jitter - np.min(eig) + 1e-12)
        self.cov  = cov
        self.corr = self._cov_to_corr(cov)
        return self

    def corr_matrix(self) -> np.ndarray:
        if self.corr is None:
            raise RuntimeError("Copula not fitted.")
        return self.corr.copy()

    def _cov_from_corr(self, corr: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.clip(np.diag(self.cov), 1e-12, None))
        D = np.diag(std)
        cov = D @ corr @ D
        cov = (cov + cov.T) / 2.0
        eig = np.linalg.eigvalsh(cov)
        if np.any(eig <= 0):
            cov = cov + np.eye(cov.shape[0]) * (self.jitter - np.min(eig) + 1e-12)
        return cov

    def sample(self, n: int, corr_override: Optional[np.ndarray] = None, **_) -> pd.DataFrame:
        if self.mean is None or self.cov is None:
            raise RuntimeError("Copula not fitted.")
        cov = self.cov if corr_override is None else self._cov_from_corr(np.asarray(corr_override))
        X = self.rng.multivariate_normal(self.mean, cov, size=int(n), method="eigh")
        return pd.DataFrame(X, columns=self.cols)
