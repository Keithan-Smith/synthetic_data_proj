import pandas as pd
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
