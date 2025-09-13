from __future__ import annotations

import os
import math
import json
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

# optional SHAP
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# optional SciPy KDE + normal quantile
try:
    from scipy.stats import gaussian_kde, norm  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ------------------------------ utilities ------------------------------

def _make_ohe():
    # sklearn >= 1.4 uses 'sparse_output'; older uses 'sparse'
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(np.float64)

def _safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _wilson_ci(phat: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    if _HAS_SCIPY:
        z = float(norm.ppf(1 - alpha / 2))
    else:
        z = 1.96  # 95%
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z/denom) * math.sqrt((phat*(1-phat))/n + (z**2)/(4*n**2))
    return (max(0.0, center - half), min(1.0, center + half))

def _logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------ correlations ---------------------------

def correlation_panels(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    cont_cols: Iterable[str],
    out_dir: str,
    title_suffix: str = "",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cont = _safe_cols(df_real, cont_cols)
    if len(cont) < 2:
        # Not enough to form a correlation panel
        return None, None, None

    R = df_real[cont].apply(_to_float_series).corr()
    S = df_synth[cont].apply(_to_float_series).corr()
    D = S - R

    def _heat(mat: pd.DataFrame, title: str, name: str):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(mat.values, vmin=-1, vmax=1, cmap="viridis")
        plt.xticks(range(len(mat.columns)), mat.columns, rotation=90)
        plt.yticks(range(len(mat.index)), mat.index)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), dpi=150)
        plt.close()

    _heat(R, f"Real corr{title_suffix}", "corr_real.png")
    _heat(S, f"Synth corr{title_suffix}", "corr_synth.png")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(D.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(D.columns)), D.columns, rotation=90)
    plt.yticks(range(len(D.index)), D.index)
    plt.title(f"Corr diff (synth - real){title_suffix}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_diff.png"), dpi=150)
    plt.close()

    return R, S, D


# ------------------------------ hist / KDE overlays --------------------

def hist_kde_overlays(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    cont_cols: Iterable[str],
    out_dir: str,
    bins: int = 40,
) -> None:
    cont = _safe_cols(df_real, cont_cols)
    for c in cont:
        r = _to_float_series(df_real[c]).dropna()
        s = _to_float_series(df_synth[c]).dropna()

        plt.figure(figsize=(6, 4))
        plt.hist(r, bins=bins, density=True, alpha=0.35, label="real")
        plt.hist(s, bins=bins, density=True, alpha=0.35, label="synth")

        if _HAS_SCIPY and len(r) > 5 and len(s) > 5:
            xs = np.linspace(np.nanmin([r.min(), s.min()]), np.nanmax([r.max(), s.max()]), 300)
            try:
                kde_r = gaussian_kde(r)
                kde_s = gaussian_kde(s)
                plt.plot(xs, kde_r(xs))
                plt.plot(xs, kde_s(xs))
            except Exception:
                pass

        plt.title(f"{c}: real vs synth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{c}.png"), dpi=150)
        plt.close()


# ------------------------------ PCA overlay ---------------------------

def pca_overlay(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    cont_cols: Iterable[str],
    out_dir: str,
    n_components: int = 2,
) -> None:
    cont = _safe_cols(df_real, cont_cols)
    if len(cont) < 2:
        return

    scaler = StandardScaler().fit(df_real[cont].apply(_to_float_series))
    Xr = scaler.transform(df_real[cont].apply(_to_float_series))
    Xs = scaler.transform(df_synth[cont].apply(_to_float_series))

    pca = PCA(n_components=n_components, random_state=0).fit(Xr)
    Zr = pca.transform(Xr)
    Zs = pca.transform(Xs)

    plt.figure(figsize=(7, 5))
    plt.scatter(Zr[:, 0], Zr[:, 1], s=12, alpha=0.5, label="real")
    plt.scatter(Zs[:, 0], Zs[:, 1], s=12, alpha=0.5, label="synth")
    plt.title("PCA overlay (continuous)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_overlay.png"), dpi=150)
    plt.close()


# ------------------------------ Decile table ---------------------------

def _decile_lift_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]

    k = max(1, n // 10)
    rows = []
    total_pos = y.sum() if n else 0
    for d in range(10):
        start = d * k
        end = n if d == 9 else (d + 1) * k
        seg = y[start:end]
        seg_n = len(seg)
        seg_pos = seg.sum()
        cap = seg_pos / total_pos if total_pos > 0 else np.nan
        rows.append({
            "decile": d + 1,
            "n": seg_n,
            "positives": int(seg_pos),
            "capture": float(cap),
            "score_min": float(s[end-1]) if seg_n > 0 else np.nan,
            "score_max": float(s[start]) if seg_n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------ CAP curve / AR / Gini -----------------

def cap_curve_and_ar(
    y_true: pd.Series,
    pd_scores: pd.Series,
    out_dir: str,
    name: str = "cap",
    annotate: bool = True,
) -> Dict[str, float]:
    _ensure_dir(out_dir)

    # clean + coerce
    y_raw = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).values
    s_raw = pd.to_numeric(pd_scores, errors="coerce").fillna(0.0).astype(float).values
    m = np.isfinite(y_raw) & np.isfinite(s_raw)
    y_raw, s_raw = y_raw[m], s_raw[m]

    n = int(len(y_raw))
    if n == 0:
        raise ValueError("cap_curve_and_ar: empty after cleaning.")

    pos = int(y_raw.sum()); neg = n - pos
    p = (pos / n) if n else 0.0

    # AUC/Gini BEFORE sorting
    try:
        auc = float(roc_auc_score(y_raw, s_raw)) if pos > 0 and neg > 0 else float("nan")
    except Exception:
        auc = float("nan")
    gini = float(2 * auc - 1) if np.isfinite(auc) else float("nan")

    # CAP sorted by score desc
    order = np.argsort(-s_raw)
    y = y_raw[order]
    frac_pop = np.arange(1, n + 1) / n
    cum_pos = np.cumsum(y)
    total_pos = pos
    frac_cap = (cum_pos / total_pos) if total_pos > 0 else np.zeros_like(cum_pos, dtype=float)

    area_cap = float(np.trapz(frac_cap, frac_pop))
    area_rand = 0.5
    area_perf = 1.0 - 0.5 * p
    ar = float((area_cap - area_rand) / (area_perf - area_rand)) if area_perf > area_rand else float("nan")

    # Top decile capture
    dsize = max(1, int(np.floor(0.10 * n)))
    top_decile_capture = float(y[:dsize].sum() / total_pos) if total_pos > 0 else float("nan")

    # Decile table
    dec_tbl = _decile_lift_table(y_raw, s_raw)
    dec_tbl.to_csv(os.path.join(out_dir, f"{name}_deciles.csv"), index=False)

    # Plot
    try:
        plt.figure(figsize=(6, 5))
        plt.plot(frac_pop, frac_cap, label=f"CAP (AR={ar:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        x_star = p
        plt.plot([0, x_star, 1], [0, 1, 1], linestyle=":", label="Perfect")
        plt.xlabel("Fraction of accounts")
        plt.ylabel("Fraction of defaults captured")
        plt.title("CAP curve")
        plt.legend()

        if annotate:
            txt = (
                f"AR={ar:.3f}\n"
                f"AUC={auc:.3f}\n"
                f"Gini={gini:.3f}\n"
                f"DR={p:.3%}\n"
                f"Top10%={top_decile_capture:.3%}\n"
                f"N={n}, Pos={pos}"
            )
            plt.gca().text(
                0.02, 0.25, txt,
                transform=plt.gca().transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.15)
            )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return {
        "AR": ar,
        "AUC": auc,
        "Gini": gini,
        "DefaultRate": float(p),
        "TopDecileCapture": top_decile_capture,
        "n": n,
        "positives": pos,
    }


# ------------------------------ KS statistic --------------------------

def ks_statistic(
    y_true: pd.Series,
    scores: pd.Series,
    out_dir: str,
    name: str = "ks",
) -> Dict[str, float]:
    _ensure_dir(out_dir)

    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).values
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0).astype(float).values
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]

    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"KS": float("nan"), "threshold": float("nan")}

    order = np.argsort(s)  # ascending
    y = y[order]
    s = s[order]

    n1 = y.sum()
    n0 = len(y) - n1
    cdf1 = np.cumsum(y) / n1
    cdf0 = np.cumsum(1 - y) / n0
    diffs = np.abs(cdf1 - cdf0)
    idx = int(np.argmax(diffs))
    ks = float(diffs[idx])
    thr = float(s[idx])

    # plot
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(s, cdf1, label="Bad CDF")
        plt.plot(s, cdf0, label="Good CDF")
        plt.vlines(thr, min(cdf1[idx], cdf0[idx]), max(cdf1[idx], cdf0[idx]), colors="r", linestyles="--",
                   label=f"KS={ks:.3f} @ {thr:.3f}")
        plt.legend()
        plt.title("KS curves")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
        f.write(f"KS={ks:.6f}\nthreshold={thr:.6f}\n")

    return {"KS": ks, "threshold": thr}


# ------------------------------ Calibration (with CIs) ----------------

def calibration_plot_with_ci(
    y_true: pd.Series,
    pd_scores: pd.Series,
    out_dir: str,
    name: str = "calibration",
    n_bins: int = 10,
    alpha: float = 0.05,
) -> Dict[str, float]:
    _ensure_dir(out_dir)

    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).values
    p = pd.to_numeric(pd_scores, errors="coerce").clip(1e-12, 1-1e-12).astype(float).values
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = len(y)
    if n == 0:
        return {"ECE": np.nan, "Brier": np.nan, "slope": np.nan, "intercept": np.nan}

    # quantile bins (equal count) for stability
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, q)
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)

    bins = np.digitize(p, edges[1:-1], right=True)
    bin_df = pd.DataFrame({"y": y, "p": p, "bin": bins})

    rows = []
    ece = 0.0
    for b in range(len(edges) - 1):
        sub = bin_df[bin_df["bin"] == b]
        nb = len(sub)
        if nb == 0:
            rows.append({"bin": b+1, "n": 0, "p_hat": np.nan, "y_bar": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        p_hat = float(sub["p"].mean())
        y_bar = float(sub["y"].mean())
        lo, hi = _wilson_ci(y_bar, nb, alpha)
        rows.append({"bin": b+1, "n": nb, "p_hat": p_hat, "y_bar": y_bar, "ci_low": lo, "ci_high": hi})
        ece += (nb / n) * abs(y_bar - p_hat)

    cal_tbl = pd.DataFrame(rows)
    cal_tbl.to_csv(os.path.join(out_dir, f"{name}_bins.csv"), index=False)

    # slope/intercept by regressing y on logit(p)
    z = _logit_clip(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(z, y)
        slope = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
    except Exception:
        slope, intercept = np.nan, np.nan

    brier = float(brier_score_loss(y, p))

    # plot
    try:
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], "--", label="Perfect")
        plt.errorbar(cal_tbl["p_hat"], cal_tbl["y_bar"],
                     yerr=[cal_tbl["y_bar"] - cal_tbl["ci_low"], cal_tbl["ci_high"] - cal_tbl["y_bar"]],
                     fmt="o", capsize=3, label="Observed (95% CI)")
        plt.xlabel("Predicted PD")
        plt.ylabel("Observed default rate")
        plt.title("Calibration (reliability) plot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    with open(os.path.join(out_dir, f"{name}_summary.txt"), "w") as f:
        f.write(f"ECE={ece:.6f}\nBrier={brier:.6f}\nslope={slope:.6f}\nintercept={intercept:.6f}\n")

    return {"ECE": ece, "Brier": brier, "slope": slope, "intercept": intercept}


# ------------------------------ PD calibration to a target  ------------

def calibrate_pd_to_target(pd_series: pd.Series, target: float = 0.02, tol: float = 1e-9) -> Tuple[pd.Series, float]:
    p = pd_series.clip(1e-9, 1 - 1e-9).astype(float).values

    def mean_shift(b: float) -> float:
        z = _logit_clip(p) + b
        pp = _sigmoid(z)
        return float(pp.mean())

    lo, hi = -20.0, 20.0
    b = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        m = mean_shift(mid)
        if abs(m - target) < tol:
            b = mid
            break
        if m < target:
            lo = mid
        else:
            hi = mid
    else:
        b = mid

    z = _logit_clip(p) + b
    p_new = _sigmoid(z)
    return pd.Series(p_new, index=pd_series.index), float(b)


# ------------------------------ PD contributions (logit) ---------------

def pd_contributions(
    df: pd.DataFrame,
    target_col: str,
    cont_cols: Iterable[str],
    cat_cols: Iterable[str],
    out_dir: str,
    model: Optional[LogisticRegression] = None,
    model_name: str = "logit_synth",
) -> Dict[str, pd.DataFrame]:
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values

    cont = _safe_cols(df, cont_cols)
    cat = _safe_cols(df, cat_cols)

    X_cont = df[cont].apply(_to_float_series) if cont else pd.DataFrame(index=df.index)
    enc = _make_ohe()
    X_cat = pd.DataFrame(index=df.index)
    if cat:
        Xc = enc.fit_transform(df[cat].astype(str))
        cat_feature_names = list(enc.get_feature_names_out(cat))
        X_cat = pd.DataFrame(Xc, index=df.index, columns=cat_feature_names)
    else:
        cat_feature_names = []

    X = pd.concat([X_cont, X_cat], axis=1)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = scaler.transform(X)

    if model is None:
        model = LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced")
        model.fit(Xs, y)

    coefs = pd.Series(model.coef_.ravel(), index=X.columns)
    odds = np.exp(coefs)
    coef_tbl = pd.DataFrame({"coef": coefs, "odds_ratio": odds}).sort_values("coef", ascending=False)
    coef_tbl.to_csv(os.path.join(out_dir, f"{model_name}_coefficients.csv"), index=True)

    perm = permutation_importance(model, Xs, y, n_repeats=10, random_state=0)
    perm_tbl = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean,
                             "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    perm_tbl.to_csv(os.path.join(out_dir, f"{model_name}_perm_importance.csv"), index=False)

    if _HAS_SHAP:
        try:
            explainer = shap.LinearExplainer(model, Xs, feature_perturbation="interventional")
            sv = explainer.shap_values(Xs)
            shap_df = pd.DataFrame(sv, columns=X.columns)
            shap_df.abs().mean().sort_values(ascending=False).to_csv(
                os.path.join(out_dir, f"{model_name}_shap_mean_abs.csv")
            )
            shap.summary_plot(sv, X, show=False, plot_type="bar", max_display=30)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_shap_bar.png"), dpi=150)
            plt.close()
        except Exception:
            pass

    return {"coefficients": coef_tbl, "permutation_importance": perm_tbl}


# ------------------------------ Helper: ROC & PR plots -----------------

def _save_roc_pr(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str) -> Dict[str, float]:
    """Save ROC and PR plots and return {'AUC':..., 'AP':...}."""
    out = {}
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, scores):.3f}")
        plt.plot([0, 1], [0, 1], "--", label="Random")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=150)
        plt.close()
        out["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["AUC"] = float("nan")

    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), dpi=150)
        plt.close()
        out["AP"] = float(ap)
    except Exception:
        out["AP"] = float("nan")

    return out


# ------------------------------ Nonlinear challenger (GBM) ------------

def challenger_gbm_synth_to_real(
    df_synth: pd.DataFrame,
    df_real: pd.DataFrame,
    target_col: str,
    cont_cols: Iterable[str],
    cat_cols: Iterable[str],
    out_dir: str,
    name: str = "gbm",
) -> Dict[str, float]:
    _ensure_dir(out_dir)

    cont = _safe_cols(df_synth, cont_cols)
    cat = _safe_cols(df_synth, cat_cols)

    # Train on synth
    y_tr = pd.to_numeric(df_synth[target_col], errors="coerce").fillna(0).astype(int).values
    Xtr_cont = df_synth[cont].apply(_to_float_series) if cont else pd.DataFrame(index=df_synth.index)
    enc = _make_ohe()
    if cat:
        Xtr_cat = enc.fit_transform(df_synth[cat].astype(str))
        cat_names = list(enc.get_feature_names_out(cat))
        Xtr_cat = pd.DataFrame(Xtr_cat, index=df_synth.index, columns=cat_names)
    else:
        Xtr_cat = pd.DataFrame(index=df_synth.index); cat_names = []
    Xtr = pd.concat([Xtr_cont, Xtr_cat], axis=1).fillna(0.0).to_numpy()

    # Test on real (same encoder for cats)
    y_te = pd.to_numeric(df_real[target_col], errors="coerce").fillna(0).astype(int).values
    Xte_cont = df_real[cont].apply(_to_float_series) if cont else pd.DataFrame(index=df_real.index)
    if cat:
        Xte_cat = enc.transform(df_real[cat].astype(str))
        Xte_cat = pd.DataFrame(Xte_cat, index=df_real.index, columns=cat_names)
    else:
        Xte_cat = pd.DataFrame(index=df_real.index)
    Xte = pd.concat([Xte_cont, Xte_cat], axis=1).fillna(0.0).to_numpy()

    gbm = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0
    )
    gbm.fit(Xtr, y_tr)

    # Scores on real
    s_real = gbm.predict_proba(Xte)[:, 1]
    try:
        auc = float(roc_auc_score(y_te, s_real)) if y_te.sum() > 0 and y_te.sum() < len(y_te) else float("nan")
    except Exception:
        auc = float("nan")

    # --- SAVE SCORES FOR DELONG ---
    scores_path = os.path.join(out_dir, f"{name}_scores_real.csv")
    pd.DataFrame({"y": y_te.astype(int), "score": s_real.astype(float)}).to_csv(scores_path, index=False)

    # CAP/AR, KS and ROC/PR
    cap = cap_curve_and_ar(pd.Series(y_te), pd.Series(s_real), out_dir, name=f"{name}_cap", annotate=True)
    ks  = ks_statistic(pd.Series(y_te), pd.Series(s_real), out_dir, name=f"{name}_ks")
    rp  = _save_roc_pr(y_te, s_real, out_dir, prefix=f"{name}")

    feats = list(Xtr_cont.columns) + cat_names
    imp = pd.DataFrame({"feature": feats, "importance": gbm.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(os.path.join(out_dir, f"{name}_importances.csv"), index=False)

    metrics = {
        "AUC_real": auc,
        "AP_real": rp.get("AP", np.nan),
        "AR_real": cap.get("AR", np.nan),
        "KS_real": ks.get("KS", np.nan)
    }
    # Also write tidy CSV for convenience
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"{name}_metrics.csv"), index=False)

    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ------------------------------ One-shot driver -----------------------

def evaluate_synth_vs_real(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    cont_cols: Iterable[str],
    cat_cols: Iterable[str],
    default_col: str,
    pd_col: str,
    out_dir: str,
    disperse_with_quantiles: bool = False,
    calib_bins: int = 10,
    calib_alpha: float = 0.05,
) -> Dict[str, object]:
    _ensure_dir(out_dir)

    # optional dispersion fix
    def _quantile_match_columns(df_synth_local: pd.DataFrame, df_real_local: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        out = df_synth_local.copy()
        for c in _safe_cols(df_real_local, cols):
            r = pd.to_numeric(df_real_local[c], errors="coerce").dropna().values
            s = pd.to_numeric(df_synth_local[c], errors="coerce").values
            if len(r) < 10:
                continue
            ranks = pd.Series(s).rank(method="average", pct=True).values
            rq = np.quantile(r, np.linspace(0, 1, 1001))
            idx = np.clip((ranks * 1000).astype(int), 0, 1000)
            out[c] = rq[idx]
        return out

    if disperse_with_quantiles:
        df_synth = _quantile_match_columns(df_synth, df_real, cont_cols)

    # correlations / distributions / PCA
    try:
        R, S, D = correlation_panels(df_real, df_synth, cont_cols, out_dir)
    except Exception:
        R, S, D = None, None, None
    hist_kde_overlays(df_real, df_synth, cont_cols, out_dir)
    pca_overlay(df_real, df_synth, cont_cols, out_dir)

    # --- Real set: CAP/KS/Calibration on provided PD column (if present) ---
    cap_metrics_real = {}
    ks_metrics_real = {}
    calib_metrics_real = {}
    if (default_col in df_real.columns) and (pd_col in df_real.columns):
        cap_metrics_real = cap_curve_and_ar(df_real[default_col], df_real[pd_col], out_dir, name="cap_real")
        ks_metrics_real  = ks_statistic(df_real[default_col], df_real[pd_col], out_dir, name="ks_real")
        calib_metrics_real = calibration_plot_with_ci(
            df_real[default_col], df_real[pd_col], out_dir, name="calibration_real",
            n_bins=calib_bins, alpha=calib_alpha
        )

    # --- Synth set: mirror same plots if columns exist ---
    cap_metrics_synth = {}
    ks_metrics_synth = {}
    calib_metrics_synth = {}
    if (default_col in df_synth.columns) and (pd_col in df_synth.columns):
        cap_metrics_synth = cap_curve_and_ar(df_synth[default_col], df_synth[pd_col], out_dir, name="cap_synth")
        ks_metrics_synth  = ks_statistic(df_synth[default_col], df_synth[pd_col], out_dir, name="ks_synth")
        calib_metrics_synth = calibration_plot_with_ci(
            df_synth[default_col], df_synth[pd_col], out_dir, name="calibration_synth",
            n_bins=calib_bins, alpha=calib_alpha
        )

    # --- Consolidated metric tables (nice for reports) ---
    try:
        # CAP
        rows = []
        if cap_metrics_real:
            rows.append(dict(dataset="real", **cap_metrics_real))
        if cap_metrics_synth:
            rows.append(dict(dataset="synth", **cap_metrics_synth))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "cap_metrics.csv"), index=False)

        # KS
        rows = []
        if ks_metrics_real:
            rows.append(dict(dataset="real", **ks_metrics_real))
        if ks_metrics_synth:
            rows.append(dict(dataset="synth", **ks_metrics_synth))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ks_metrics.csv"), index=False)

        # Calibration summary
        rows = []
        if calib_metrics_real:
            rows.append(dict(dataset="real", **calib_metrics_real))
        if calib_metrics_synth:
            rows.append(dict(dataset="synth", **calib_metrics_synth))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)
    except Exception:
        pass

    # --- PD contributions (logistic on synthetic) ---
    contrib = {}
    if default_col in df_synth.columns:
        contrib = pd_contributions(
            df=df_synth.dropna(subset=[default_col]),
            target_col=default_col,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            out_dir=out_dir,
            model=None,
            model_name="pd_on_synth",
        )

    # --- Nonlinear challenger: GBM trained on synth, tested on real ---
    challenger = {}
    if (default_col in df_synth.columns) and (default_col in df_real.columns):
        challenger = challenger_gbm_synth_to_real(
            df_synth=df_synth, df_real=df_real,
            target_col=default_col,
            cont_cols=cont_cols, cat_cols=cat_cols,
            out_dir=out_dir, name="gbm"
        )

    # --- Example: calibrate synthetic PDs to 2% mean (write CSV + txt) ---
    if pd_col in df_synth.columns:
        cal_pd, shift = calibrate_pd_to_target(df_synth[pd_col], target=0.02)
        cal_pd.to_frame("pd_calibrated").to_csv(os.path.join(out_dir, "pd_calibrated_2pct.csv"), index=False)
        with open(os.path.join(out_dir, "pd_calibration.txt"), "w") as f:
            f.write(f"intercept_shift_b = {shift:.6f}\n")
            f.write(f"original_mean_pd = {df_synth[pd_col].mean():.6f}\n")
            f.write(f"calibrated_mean_pd = {cal_pd.mean():.6f}\n")

    return {
        "corr_real": R, "corr_synth": S, "corr_diff": D,
        "cap_metrics_real": cap_metrics_real,
        "cap_metrics_synth": cap_metrics_synth,
        "ks_real": ks_metrics_real,
        "ks_synth": ks_metrics_synth,
        "calibration_real": calib_metrics_real,
        "calibration_synth": calib_metrics_synth,
        "pd_contributions": contrib,
        "challenger_gbm": challenger,
    }
