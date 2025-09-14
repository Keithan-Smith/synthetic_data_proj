import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_histograms(df_real, df_syn, cols, out_dir, bins=30):
    import matplotlib.pyplot as plt
    for c in cols:
        plt.figure()
        df_real[c].dropna().plot(kind="hist", bins=bins, density=True, alpha=0.5, label="real")
        df_syn[c].dropna().plot(kind="hist", bins=bins, density=True, alpha=0.5, label="synth")
        plt.title(f"Histogram: {c}")
        plt.xlabel(c); plt.ylabel("density")
        plt.legend()
        plt.savefig(f"{out_dir}/hist_{c}.png", bbox_inches="tight")
        plt.close()

def plot_pca(real: pd.DataFrame, synth: pd.DataFrame, cols, path):
    pca = PCA(n_components=2)
    r = real[cols].dropna()
    s = synth[cols].dropna()
    n = min(len(r), len(s), 2000)
    r = r.sample(n) if len(r)>n else r
    s = s.sample(n) if len(s)>n else s
    if len(r)==0 or len(s)==0: return
    import numpy as np
    X = pd.concat([r, s], axis=0).to_numpy()
    Z = pca.fit_transform(X)
    plt.figure()
    plt.scatter(Z[:len(r),0], Z[:len(r),1], alpha=0.5, label="real", s=8)
    plt.scatter(Z[len(r):,0], Z[len(r):,1], alpha=0.5, label="synth", s=8)
    plt.legend(); plt.title("PCA overlay (cont features)")
    plt.savefig(f"{path}/pca_overlay.png"); plt.close()

def _corr_matrix(df: pd.DataFrame, cols: list, method: str = "pearson") -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            # try to coerce
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.corr(method=method)

def plot_corr_compare(real_df: pd.DataFrame,
                      synth_df: pd.DataFrame,
                      cols: list,
                      out_dir: str,
                      method: str = "pearson",
                      fname_prefix: str = "corr") -> dict:
    """
    Saves three heatmaps:
      - {prefix}_real.png
      - {prefix}_synth.png
      - {prefix}_absdiff.png   (|real - synth|)
    Returns summary stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in cols if c in real_df.columns and c in synth_df.columns]
    if len(cols) < 2:
        return {"used_cols": cols, "note": "Not enough common columns for correlation."}

    C_real = _corr_matrix(real_df, cols, method=method).fillna(0.0)
    C_synth = _corr_matrix(synth_df, cols, method=method).fillna(0.0)
    C_diff = (C_real - C_synth).abs()

    def _heat(M: pd.DataFrame, title: str, path: str, vmin=-1.0, vmax=1.0, cmap="coolwarm"):
        plt.figure(figsize=(max(6, 0.6*len(cols)), max(5, 0.6*len(cols))))
        im = plt.imshow(M.values, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(title)
        plt.xticks(range(len(cols)), cols, rotation=90)
        plt.yticks(range(len(cols)), cols)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    _heat(C_real,  f"Real correlations ({method})",  os.path.join(out_dir, f"{fname_prefix}_real.png"))
    _heat(C_synth, f"Synth correlations ({method})", os.path.join(out_dir, f"{fname_prefix}_synth.png"))
    _heat(C_diff,  f"|Real−Synth| correlation diff", os.path.join(out_dir, f"{fname_prefix}_absdiff.png"),
          vmin=0.0, vmax=1.0, cmap="viridis")

    return {
        "used_cols": cols,
        "avg_abs_diff": float(np.mean(np.abs(C_real.values - C_synth.values))),
        "max_abs_diff": float(np.max(np.abs(C_real.values - C_synth.values))),
    }