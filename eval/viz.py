import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_histograms(real: pd.DataFrame, synth: pd.DataFrame, cols, path):
    for c in cols:
        if c not in real.columns or c not in synth.columns: continue
        plt.figure()
        real[c].dropna().plot(kind="hist", alpha=0.5, density=True)
        synth[c].dropna().plot(kind="hist", alpha=0.5, density=True)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c); plt.ylabel("density")
        plt.savefig(f"{path}/hist_{c}.png"); plt.close()

def plot_corr(real: pd.DataFrame, synth: pd.DataFrame, cols, path):
    import numpy as np
    plt.figure()
    plt.imshow(real[cols].corr(), aspect='auto'); plt.title("Real corr")
    plt.colorbar(); plt.savefig(f"{path}/corr_real.png"); plt.close()
    plt.figure()
    plt.imshow(synth[cols].corr(), aspect='auto'); plt.title("Synth corr")
    plt.colorbar(); plt.savefig(f"{path}/corr_synth.png"); plt.close()

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
