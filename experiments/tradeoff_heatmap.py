import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "outputs/credit_run/credit_eval/tradeoff_results.csv"
OUT = os.path.dirname(CSV) or "."

df = pd.read_csv(CSV)

def _heat(df, value_col, fname, title):
    pv = df.pivot_table(index="dp_noise_multiplier",
                        columns="privreg_lambda_gan",
                        values=value_col, aggfunc="mean")
    pv = pv.sort_index().reindex(sorted(pv.columns), axis=1)
    plt.figure(figsize=(6,4.6))
    im = plt.imshow(pv.values, aspect="auto")
    plt.xticks(range(pv.shape[1]), [f"{c:g}" for c in pv.columns], rotation=0)
    plt.yticks(range(pv.shape[0]), [f"{r:g}" for r in pv.index])
    plt.xlabel("MMD λ (GAN)")
    plt.ylabel("DP noise μ")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # overlay values
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()

if "auc" in df.columns:
    _heat(df, "auc", "grid_auc.png", "AUC (synth→real)")
if "mia_auc" in df.columns:
    _heat(df, "mia_auc", "grid_mia_auc.png", "MIA-AUC (lower = better privacy)")



