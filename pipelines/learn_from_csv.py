"""
Lean orchestrator:
  1) Load data (CSV via path OR UCI via ucimlrepo id)
  2) Infer schema, pick cont/cat (exclude targets)
  3) Train HybridGenerator (Copula+VAE+GAN+Cat-AR)
  4) Sample synthetic data (optionally with ShockSpec from YAML)
  5) (optional) credit portfolio simulation + PD calibration (2% avg default)
  6) Eval (fidelity, utility, privacy, credit), visuals, HTML report
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif, chi2

from utils.config import load_config
from data.adapters.universal_csv import load_any_csv, normalize_df
from data.schemas import infer_schema_from_df
from hybrid.hybrid import HybridGenerator
from shocks.spec import ShockSpec

from eval.eval_credit import evaluate_synth_vs_real
from eval.fidelity import univariate_fidelity, correlation_diff
from eval.privacy import membership_inference_auc, distance_to_closest_record
from eval.utility import generic_binary_downstream_eval
from eval.viz import plot_histograms, plot_pca, plot_corr_compare
from eval.report import build_report


# ---------------- small helpers ----------------
def _maybe_credit_bits():
    try:
        from simulator.portfolio import simulate_portfolio
        from simulator.pd_calibration import PDCalibrator
        return simulate_portfolio, PDCalibrator
    except Exception:
        return None, None

def _resolve_with_cfg(base_dir, pathlike):
    if not pathlike:
        return ""
    if os.path.isabs(pathlike):
        return pathlike
    return os.path.join(base_dir or "", pathlike)

def _parse_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def _parse_shock_from_cfg(cfg) -> ShockSpec:
    s = getattr(cfg, "shock", None) or {}
    if not isinstance(s, dict):
        return ShockSpec()
    pairs = {}
    raw_pairs = s.get("corr_pairs", {})
    if isinstance(raw_pairs, list):
        for trip in raw_pairs:
            if isinstance(trip, (list, tuple)) and len(trip) == 3:
                a, b, rho = trip
                pairs[(str(a), str(b))] = float(rho)
    elif isinstance(raw_pairs, dict):
        for k, v in raw_pairs.items():
            if isinstance(k, str):
                k2 = k.strip().strip("()")
                parts = [p.strip() for p in k2.split(",")]
                if len(parts) == 2:
                    pairs[(parts[0], parts[1])] = float(v)
    s = dict(s)
    s["corr_pairs"] = pairs
    try:
        return ShockSpec(**s).validate()
    except Exception:
        return ShockSpec()

def _load_df(cfg) -> pd.DataFrame:
    colmap = _resolve_with_cfg(getattr(cfg, "config_dir", ""), getattr(cfg, "column_map", ""))
    alias  = _resolve_with_cfg(getattr(cfg, "config_dir", ""), getattr(cfg, "domain_pack", ""))

    if str(getattr(cfg, "data_source", "")).lower() == "ucimlrepo":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as e:
            raise RuntimeError("Install ucimlrepo: pip install ucimlrepo") from e
        if not getattr(cfg, "uci_id", None):
            raise ValueError("Missing 'uci_id' for ucimlrepo source.")

        ds = fetch_ucirepo(id=int(cfg.uci_id))
        X = ds.data.features.reset_index(drop=True)
        y = ds.data.targets.reset_index(drop=True) if ds.data.targets is not None else None

        df = X.copy()
        if y is not None and y.shape[1] > 0:
            for col in y.columns:
                if col in df.columns:
                    df[f"target_{col}"] = y[col]
                else:
                    df[col] = y[col]
            # map first target to canonical if it's the German "class"
            tcol = y.columns[0]
            if tcol in df.columns:
                s = df[tcol]
                if s.dtype == object:
                    sl = s.astype(str).str.lower()
                    if set(sl.unique()) <= {"good","bad"}:
                        df["bad_within_horizon"] = (sl == "bad").astype(int)
                        df = df.drop(columns=[tcol])
                else:
                    vals = set(s.dropna().unique().tolist())
                    if vals <= {1, 2}:
                        df["bad_within_horizon"] = s.map({1: 0, 2: 1})
                        df = df.drop(columns=[tcol])

        return normalize_df(
            df,
            column_map_path=(colmap or None),
            alias_pack_path=(alias or None),
            parse_dates=True,
        )

    # CSV / URL path
    dp = getattr(cfg, "data_path", "")
    if not dp:
        raise ValueError("Config error: set data_source=ucimlrepo & uci_id, or provide data_path to a CSV.")
    dp = _resolve_with_cfg(getattr(cfg, "config_dir", ""), dp)
    return load_any_csv(
        dp,
        column_map_path=(colmap or None),
        alias_pack_path=(alias or None),
        parse_dates=True,
        read_csv_kwargs=(getattr(cfg, "read_csv_kwargs", None) or None),
    )

def _select_pd_features(df: pd.DataFrame, target_col: str,
                        max_cont=8, max_cat=8, max_cat_levels=30,
                        include=None, exclude=None):
    include = include or []
    exclude = exclude or []
    def _keep(name):
        if any(re.search(rx, name) for rx in exclude): return False
        if include and any(re.search(rx, name) for rx in include): return True
        return True

    y = df[target_col]
    if y.dtype == object:
        yl = y.astype(str).str.lower()
        if set(yl.unique()) <= {"good","bad"}: y = (yl=="bad").astype(int)
        elif set(yl.unique()) <= {"yes","no"}: y = (yl=="yes").astype(int)
        elif set(yl.unique()) <= {"true","false"}: y = (yl=="true").astype(int)
    if y.nunique() != 2:
        return [], []

    cont_cands = [c for c in df.columns
                  if c!=target_col and pd.api.types.is_numeric_dtype(df[c]) and _keep(c)]
    cat_cands  = [c for c in df.columns
                  if c!=target_col and not pd.api.types.is_numeric_dtype(df[c]) and _keep(c)]

    cont_scores = []
    if cont_cands:
        Xc = df[cont_cands].copy()
        Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(Xc.median(numeric_only=True))
        try:
            mi = mutual_info_classif(Xc.to_numpy(), y.to_numpy(), discrete_features=False, random_state=42)
        except Exception:
            mi = np.zeros(len(cont_cands))
        cont_scores = sorted(zip(cont_cands, mi), key=lambda t: t[1], reverse=True)

    cat_scores = []
    for c in cat_cands:
        vc = df[c].astype(str).value_counts(dropna=False)
        if len(vc) > max_cat_levels:
            continue
        X = pd.get_dummies(df[c].astype(str), drop_first=False)
        try:
            stat, _ = chi2(X, y)
            cat_scores.append((c, float(np.nan_to_num(stat).sum())))
        except Exception:
            cat_scores.append((c, 0.0))
    cat_scores = sorted(cat_scores, key=lambda t: t[1], reverse=True)

    sel_cont = [c for c,_ in cont_scores[:max_cont]]
    sel_cat  = [c for c,_ in cat_scores[:max_cat]]
    return sel_cont, sel_cat

def _coerce_binary_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        sl = s.astype(str).str.lower()
        if set(sl.unique()) <= {"good","bad"}:   return (sl == "bad").astype(int)
        if set(sl.unique()) <= {"yes","no"}:     return (sl == "yes").astype(int)
        if set(sl.unique()) <= {"true","false"}: return (sl == "true").astype(int)
    vals = set(pd.unique(s))
    if vals <= {1,2}:
        return s.map({1:0, 2:1})
    return s.astype(int)

def _ensure_pd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical PD columns exist:
      - bad_within_horizon (binary 0/1)
      - default_flag      (alias of bad_within_horizon if missing)
      - pd_monthly / pd_12m (derive one from the other if needed)
    """
    if "bad_within_horizon" not in df.columns:
        cand = next((c for c in ["default_flag","default","class","label","target"] if c in df.columns), None)
        if cand is not None:
            df["bad_within_horizon"] = _coerce_binary_series(df[cand])
    if "default_flag" not in df.columns and "bad_within_horizon" in df.columns:
        df["default_flag"] = df["bad_within_horizon"].astype(int)

    has_m = "pd_monthly" in df.columns
    has_12 = "pd_12m" in df.columns
    if not has_m and "pd" in df.columns:
        df.rename(columns={"pd": "pd_12m"}, inplace=True)
        has_12 = True
    if has_m and not has_12:
        df["pd_12m"] = 1.0 - (1.0 - df["pd_monthly"].clip(0,1))**12
    if has_12 and not has_m:
        df["pd_monthly"] = 1.0 - (1.0 - df["pd_12m"].clip(0,1))**(1/12)
    return df

def _logit(x): x = np.clip(x, 1e-12, 1-1e-12); return np.log(x/(1-x))
def _sigm(z): return 1/(1+np.exp(-z))

def _calibrate_mean_rate(pd_series: pd.Series, target_rate: float):
    p = pd_series.astype(float).values
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return pd_series, 0.0
    z = _logit(p[mask])
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        m = _sigm(z + mid).mean()
        if m < target_rate: lo = mid
        else: hi = mid
    b = 0.5*(lo+hi)
    shifted = pd.Series(p, index=pd_series.index)
    shifted.loc[mask] = _sigm(z + b)
    return shifted, float(b)

def _max_epsilon_from_log(train_eps_log: dict|None) -> float|None:
    if not train_eps_log:
        return None
    vals = []
    for v in train_eps_log.values():
        try:
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        except Exception:
            pass
    return max(vals) if vals else None

def _tradeoff_metrics(hg, df_real, syn_o, cont_cols) -> dict:
    util = generic_binary_downstream_eval(train_df=syn_o, test_df=df_real, target_col='bad_within_horizon',
                                          cont_cols=cont_cols, cat_cols=None, 
                                          return_scores = True)
    auc = util.get("auc")
    ar = (2*auc - 1.0) if (auc is not None and np.isfinite(auc)) else None
    mia = membership_inference_auc(df_real, syn_o, cont_cols)
    eps = _max_epsilon_from_log(getattr(hg, "train_eps_log", None))
    return {"auc": auc, "ar": ar, "mia_auc": mia, "epsilon": eps}

def _plot_tradeoff(xvals, yvals, xlabel, ylabel, out_path, title=None):
    plt.figure(figsize=(6,4))
    plt.plot(xvals, yvals, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def _sweep_privacy_utility(cfg, df, schema, cont_cols, cat_cols, base_privacy_cfg, out_dir):
    """
    Optional sweeps over dp_noise_multiplier and/or privreg_lambda_gan.
    """
    os.makedirs(out_dir, exist_ok=True)

    sweep = getattr(cfg, "sweep", None) or {}
    dp_list = sweep.get("dp_noise_list") or []
    lam_list = sweep.get("mmd_lambda_gan_list") or []
    mmd_mode = str(sweep.get("mmd_mode", getattr(cfg, "privreg_mode", "repulsive"))).lower()
    lam_vae_factor = float(sweep.get("mmd_lambda_vae_factor", 0.25))
    do_grid = bool(sweep.get("grid", False))

    if not dp_list and not lam_list:
        return

    rows = []

    def _one_run(dp_noise=None, lam_gan=None):
        p_cfg = dict(base_privacy_cfg)
        if dp_noise is not None:
            p_cfg["enabled"] = True
            p_cfg["dp_noise_multiplier"] = float(dp_noise)

        privreg_cfg = {
            "enabled": (lam_gan is not None) and (float(lam_gan) > 0.0),
            "mode": mmd_mode,
            "lambda_mmd_gan": float(lam_gan) if lam_gan is not None else 0.0,
            "lambda_mmd_vae": float(lam_vae_factor) * (float(lam_gan) if lam_gan is not None else 0.0),
            "mmd_bandwidth": float(getattr(cfg, "privreg_mmd_bandwidth", 1.0)),
        }

        hg = HybridGenerator(cont_cols=cont_cols, cat_cols=cat_cols, schema=schema, device=cfg.device)
        hg.fit(df,
               epochs_vae=getattr(cfg, "vae_epochs", 20),
               epochs_gan=getattr(cfg, "gan_epochs", 100),
               batch=getattr(cfg, "batch_size", 256),
               privacy_cfg=p_cfg,
               privacy_reg_cfg=privreg_cfg,
               mine_cfg={"enabled": bool(getattr(cfg, "mine_enabled", False)),
                         "lambda": float(getattr(cfg, "mine_lambda", 0.0))})

        syn = hg.sample(len(df), shock=ShockSpec())
        if 'bad_within_horizon' not in syn.columns and 'bad_within_horizon' in df.columns:
            base = float(df['bad_within_horizon'].mean())
            syn['bad_within_horizon'] = (np.random.rand(len(syn)) < base).astype(int)

        m = _tradeoff_metrics(hg, df, syn, cont_cols)
        m["dp_noise_multiplier"] = p_cfg.get("dp_noise_multiplier")
        m["privreg_lambda_gan"] = privreg_cfg.get("lambda_mmd_gan")
        m["privreg_lambda_vae"] = privreg_cfg.get("lambda_mmd_vae")
        m["privreg_mode"] = privreg_cfg.get("mode")
        rows.append(m)

    if dp_list and lam_list and do_grid:
        for dn in dp_list:
            for lg in lam_list:
                _one_run(dp_noise=dn, lam_gan=lg)
    else:
        for dn in dp_list:
            _one_run(dp_noise=dn, lam_gan=None)
        for lg in lam_list:
            _one_run(dp_noise=None, lam_gan=lg)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "tradeoff_results.csv"), index=False)

    dp_mask = res["dp_noise_multiplier"].notna()
    if dp_mask.any() and res.loc[dp_mask, "epsilon"].notna().any():
        r = res.loc[dp_mask].sort_values("epsilon")
        r = r[np.isfinite(r["epsilon"])]
        if not r.empty:
            if r["auc"].notna().any():
                _plot_tradeoff(
                    r["epsilon"], r["auc"],
                    "ε (approx RDP)", "AUC (synth→real)",
                    os.path.join(out_dir, "tradeoff_auc_vs_epsilon.png"),
                    title="Utility vs Privacy (DP)"
                )
            if r["mia_auc"].notna().any():
                _plot_tradeoff(
                    r["epsilon"], r["mia_auc"],
                    "ε (approx RDP)", "MIA-AUC (lower=better)",
                    os.path.join(out_dir, "tradeoff_mia_vs_epsilon.png")
                )

    lam_mask = res["privreg_lambda_gan"].notna()
    if lam_mask.any():
        r = res.loc[lam_mask].sort_values("privreg_lambda_gan")
        if not r.empty:
            if r["auc"].notna().any():
                _plot_tradeoff(
                    r["privreg_lambda_gan"], r["auc"],
                    "MMD λ (GAN)", "AUC (synth→real)",
                    os.path.join(out_dir, "tradeoff_auc_vs_lambda.png"),
                    title=f"Utility vs Privacy Regularizer (mode={mmd_mode})"
                )
            if r["mia_auc"].notna().any():
                _plot_tradeoff(
                    r["privreg_lambda_gan"], r["mia_auc"],
                    "MMD λ (GAN)", "MIA-AUC (lower=better)",
                    os.path.join(out_dir, "tradeoff_mia_vs_lambda.png")
                )


# ---------------- main pipeline ----------------
def run(cfg_path: str):
    cfg = load_config(cfg_path)
    if not getattr(cfg, "config_dir", None):
        cfg.config_dir = os.path.dirname(os.path.abspath(cfg_path))
    if not getattr(cfg, "output_dir", None):
        cfg.output_dir = "outputs"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # minimal evaluation knob (metrics-only)
    eval_minimal = bool(getattr(cfg, "eval_minimal", True))

    # CUDA -> CPU fallback
    if str(getattr(cfg, "device", "cuda")).lower() == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU.")
        cfg.device = "cpu"

    # Resolve mappings once (also used to clean panel output)
    colmap_path = _resolve_with_cfg(cfg.config_dir, getattr(cfg, "column_map", ""))
    alias_path  = _resolve_with_cfg(cfg.config_dir, getattr(cfg, "domain_pack", ""))

    # 1) Load + normalize input
    df = _load_df(cfg)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded dataframe is empty. Check data_source/data_path/uci_id and mapping files.")

    # ---- Force ordinal-coded small-int fields to categorical/ordinal
    ORDINAL_CODE_COLS = [
        "installment_rate_pct",  # Attribute8
        "residence_since",       # Attribute11
        "existing_credits",      # Attribute16
        "dependents",            # Attribute18
    ]
    for c in ORDINAL_CODE_COLS:
        if c in df.columns:
            df[c] = pd.Categorical(df[c].astype("Int64"), ordered=True)

    # 2) schema + feature sets (exclude targets)
    schema = infer_schema_from_df(df.select_dtypes(exclude=["datetime", "datetimetz"]))

    PREFERRED_TARGETS = {
        "bad_within_horizon","default","default_flag","class","label","target",
        "mortality_30d","readmission_30d"
    }
    target_like = {c for c in df.columns if c in PREFERRED_TARGETS or df[c].dropna().nunique() == 2}

    cont_cols = (getattr(cfg, "cont_cols", None) or schema.cont_cols())
    cat_cols  = (getattr(cfg, "cat_cols", None)  or schema.cat_cols())

    cont_cols = [c for c in cont_cols if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]
    cat_cols  = [c for c in cat_cols if c in df.columns
                 and not pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]

    if len(cont_cols) == 0:
        numeric_present = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(
            "No continuous columns detected after filtering. "
            "Provide 'cont_cols' in YAML, or adjust mapping/aliases. "
            f"Numeric columns seen: {numeric_present[:20]} (showing up to 20). "
            f"Target-like excluded: {sorted(list(target_like))[:20]}"
        )

    # 3) Train hybrid generator
    try:
        hg = HybridGenerator(cont_cols=cont_cols, cat_cols=cat_cols, schema=schema, device=cfg.device)

        # DP settings (approx RDP)
        privacy_cfg = {
            "enabled": _parse_bool(getattr(cfg, "privacy_enabled", False)),
            "dp_max_grad_norm": float(getattr(cfg, "dp_max_grad_norm", 1.0)),
            "dp_noise_multiplier": float(getattr(cfg, "dp_noise_multiplier", 0.0)),
            "dp_delta": float(getattr(cfg, "dp_delta", 1e-5)),
        }

        # privacy–utility regularizer (MMD) — off by default for clean MINE attribution
        privacy_reg_cfg = {
            "enabled": _parse_bool(getattr(cfg, "privreg_enabled", False)),
            "mode": str(getattr(cfg, "privreg_mode", "attractive")).lower(),
            "lambda_mmd": float(getattr(cfg, "privreg_lambda", 0.0)),
            "lambda_mmd_gan": float(getattr(cfg, "privreg_lambda_gan", getattr(cfg, "privreg_lambda", 0.0))),
            "lambda_mmd_vae": float(getattr(cfg, "privreg_lambda_vae",
                                            0.25 * float(getattr(cfg, "privreg_lambda", 0.0)))),
            "mmd_bandwidth": float(getattr(cfg, "privreg_mmd_bandwidth", 1.0)),
        }

        mine_cfg = {
            "enabled": _parse_bool(getattr(cfg, "mine_enabled", False)),
            "lambda": float(getattr(cfg, "mine_lambda", 0.0)),
        }

        hg.fit(df, epochs_vae=getattr(cfg, "vae_epochs", 20),
               epochs_gan=getattr(cfg, "gan_epochs", 100),
               batch=getattr(cfg, "batch_size", 256),
               privacy_cfg=privacy_cfg,
               privacy_reg_cfg=privacy_reg_cfg,
               mine_cfg=mine_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize/train HybridGenerator. "
                           f"cont_cols={cont_cols}, cat_cols={cat_cols}. "
                           f"Details: {e}") from e

    # 3b) PDCalibrator on REAL df
    simulate_portfolio, PDCalibrator = _maybe_credit_bits()
    pdcal = None
    bin_targets = [c for c in df.columns if df[c].dropna().nunique()==2]
    if bin_targets:
        tcol = next((c for c in ["bad_within_horizon","default","default_flag","class","label","target"]
                     if c in df.columns), bin_targets[0])
        if "bad_within_horizon" not in df.columns:
            df["bad_within_horizon"] = _coerce_binary_series(df[tcol])

        if PDCalibrator is not None:
            pdcal_cfg = getattr(cfg, "pdcal", {}) or {}
            mode = str(pdcal_cfg.get("mode","auto")).lower()
            horizon = int(pdcal_cfg.get("horizon_months", 12))
            if mode == "auto":
                sel_cont, sel_cat = _select_pd_features(
                    df, "bad_within_horizon",
                    max_cont=int(pdcal_cfg.get("max_cont", 8)),
                    max_cat=int(pdcal_cfg.get("max_cat", 8)),
                    max_cat_levels=int(pdcal_cfg.get("max_cat_levels", 30)),
                    include=pdcal_cfg.get("include") or [],
                    exclude=pdcal_cfg.get("exclude") or []
                )
            else:
                sel_cont = pdcal_cfg.get("cont_cols") or []
                sel_cat  = pdcal_cfg.get("cat_cols") or []
            pdcal = PDCalibrator(cont_cols=sel_cont, cat_cols=sel_cat, horizon_months=horizon)
            pdcal.fit(df.copy(), target_col="bad_within_horizon")

    # 4) Sample synthetic originations (shocks capability kept, usually off)
    use_shock = _parse_bool(getattr(cfg, "shock_enabled", False))  # default False per study scope
    shock_spec = _parse_shock_from_cfg(cfg) if use_shock else ShockSpec()
    syn_o = hg.sample(len(df), shock=shock_spec)

    # ---- Ensure PD columns exist on BOTH real & synthetic if calibrator is available
    if pdcal is not None:
        if "pd_12m" not in df.columns:
            df["pd_12m"] = pdcal.predict_pd(df)
            df["pd_monthly"] = 1.0 - (1.0 - df["pd_12m"].clip(0,1))**(1/12)
        if "pd_12m" not in syn_o.columns:
            syn_o["pd_12m"] = pdcal.predict_pd(syn_o)
            syn_o["pd_monthly"] = 1.0 - (1.0 - syn_o["pd_12m"].clip(0,1))**(1/12)

    # Attach synthetic binary target if missing
    if 'bad_within_horizon' not in syn_o.columns:
        if pdcal is not None and "pd_12m" in syn_o.columns:
            rng = np.random.default_rng(7)
            syn_o['bad_within_horizon'] = (rng.random(len(syn_o)) < syn_o["pd_12m"].values).astype(int)
        elif 'bad_within_horizon' in df.columns:
            base = float(df['bad_within_horizon'].mean())
            syn_o['bad_within_horizon'] = (np.random.rand(len(syn_o)) < base).astype(int)

    # 5) Optional credit portfolio panel
    panel = None
    if str(getattr(cfg, "task", "none")) == "credit_portfolio" and simulate_portfolio is not None:
        panel = simulate_portfolio(
            hybrid_model=hg, originations_df=df,
            months=getattr(cfg, "months", 24),
            start_date=getattr(cfg, "start_date", "2019-01-31"),
            init_customers=getattr(cfg, "init_customers", 1000),
            base_new=getattr(cfg, "base_new", 100),
            pd_calibrator=pdcal,
            pd_logit_shift=getattr(cfg, "pd_logit_shift", 0.0),
            macro_pd_mult=getattr(cfg, "macro_pd_mult", 1.0),
            lgd_cfg=getattr(cfg, "lgd", None),
            collateral_cfg=getattr(cfg, "collateral_models", None),
            purpose_to_asset=getattr(cfg, "purpose_to_asset", None),
            mortgage_heuristic=getattr(cfg, "mortgage_heuristic", None),
        )

        try:
            panel = normalize_df(
                panel,
                column_map_path=(colmap_path or None),
                alias_pack_path=(alias_path or None),
                parse_dates=True,
            )
        except Exception:
            pass

        panel = _ensure_pd_columns(panel)

        target_rate = float(getattr(cfg, "pd_target_rate", 0.02))
        if "pd_12m" in panel.columns:
            panel["pd_12m_cal"], _ = _calibrate_mean_rate(panel["pd_12m"], target_rate)
            panel["pd_monthly_cal"] = 1.0 - (1.0 - panel["pd_12m_cal"])**(1/12)
        elif "pd_monthly" in panel.columns:
            panel["pd_monthly_cal"], _ = _calibrate_mean_rate(panel["pd_monthly"], 1.0 - (1.0 - target_rate)**(1/12))
            panel["pd_12m_cal"] = 1.0 - (1.0 - panel["pd_monthly_cal"])**12

        if "default_flag" not in panel.columns and "pd_12m_cal" in panel.columns:
            rng = np.random.default_rng(42)
            panel["default_flag"] = (rng.random(len(panel)) < panel["pd_12m_cal"].values).astype(int)

        panel.to_csv(f"{cfg.output_dir}/panel_synth.csv", index=False)

    # 6) Core evaluation (metrics-focused)
    cont_eval = [c for c in cont_cols if c != "loan_balance"]
    fid  = univariate_fidelity(df, syn_o, cont_cols)
    corr = correlation_diff(df, syn_o, cont_eval)

    # sanitize for utility
    _RESERVED = {
        "bad_within_horizon", "default_flag",
        "pd_12m", "pd_monthly", "pd_12m_cal", "pd_monthly_cal", "pd", "pd_score"
    }
    cont_for_util = [c for c in cont_cols if (c in df.columns and c in syn_o.columns and c not in _RESERVED)]
    cat_for_util  = [c for c in cat_cols if (c in df.columns and c in syn_o.columns and c not in _RESERVED)]

    util = generic_binary_downstream_eval(
        train_df=syn_o,
        test_df=df,
        target_col='bad_within_horizon',
        cont_cols=cont_for_util,
        cat_cols=cat_for_util,
    )
    # Save labels/scores for DeLong later
    np.savez(os.path.join(cfg.output_dir, "eval_scores_main.npz"),
            y=util["y_true"], p=util["scores"])
    mia  = membership_inference_auc(df, syn_o, cont_cols)
    dcr  = distance_to_closest_record(df, syn_o, cont_cols)

    # ---------- Credit evaluation (+ colocated sweep writing) ----------
    try:
        cont_ev = [c for c in cont_cols if c in df.columns]
        cat_ev  = [c for c in cat_cols if c in df.columns]

        default_col = next(
            (c for c in ["bad_within_horizon", "default_flag", "target", "class", "label"] if c in df.columns),
            None
        )
        pd_col = next(
            (c for c in ["pd_12m", "pd_monthly", "pd", "pd_score"] if c in df.columns),
            None
        )
        if pd_col is None and pdcal is not None:
            df["pd_12m"] = pdcal.predict_pd(df)
            df["pd_monthly"] = 1.0 - (1.0 - df["pd_12m"].clip(0,1))**(1/12)
            pd_col = "pd_12m"

        eval_out_dir = os.path.join(cfg.output_dir, "credit_eval")
        os.makedirs(eval_out_dir, exist_ok=True)

        evaluate_synth_vs_real(
            df_real=df,
            df_synth=syn_o,
            cont_cols=cont_ev,
            cat_cols=cat_ev,
            default_col=(default_col or "bad_within_horizon"),
            pd_col=(pd_col or "pd_12m"),
            out_dir=eval_out_dir,
            disperse_with_quantiles=getattr(cfg, "disperse_quantiles", False),
        )

        swp = getattr(cfg, "sweep", None)
        if swp and bool(swp.get("enabled", False)):
            try:
                base_privacy_cfg = {
                    "enabled": getattr(cfg, "privacy_enabled", False),
                    "dp_max_grad_norm": getattr(cfg, "dp_max_grad_norm", 1.0),
                    "dp_noise_multiplier": getattr(cfg, "dp_noise_multiplier", 0.0),
                    "dp_delta": getattr(cfg, "dp_delta", 1e-5),
                }
                _sweep_privacy_utility(
                    cfg, df, schema, cont_cols, cat_cols, base_privacy_cfg,
                    out_dir=eval_out_dir
                )
            except Exception as e:
                with open(f"{cfg.output_dir}/pipeline_warning.txt","a") as f:
                    f.write("\n[tradeoff_sweep] " + str(e))

    except Exception as e:
        with open(f"{cfg.output_dir}/pipeline_warning.txt","a") as f:
            f.write("\n[eval_credit] " + str(e))

    # ---------- Optional baselines/ablations ----------
    try:
        if _parse_bool(getattr(cfg, "run_baselines", False)):
            base_dir = os.path.join(cfg.output_dir, "baselines")
            os.makedirs(base_dir, exist_ok=True)

            hg_noreg = HybridGenerator(cont_cols=cont_cols, cat_cols=cat_cols, schema=schema, device=cfg.device)
            privacy_cfg2 = {
                "enabled": _parse_bool(getattr(cfg, "privacy_enabled", False)),
                "dp_max_grad_norm": float(getattr(cfg, "dp_max_grad_norm", 1.0)),
                "dp_noise_multiplier": float(getattr(cfg, "dp_noise_multiplier", 0.0)),
                "dp_delta": float(getattr(cfg, "dp_delta", 1e-5)),
            }
            privreg_off = {
                "enabled": False, "mode": "attractive",
                "lambda_mmd": 0.0, "lambda_mmd_gan": 0.0, "lambda_mmd_vae": 0.0, "mmd_bandwidth": 1.0,
            }
            hg_noreg.fit(
                df,
                epochs_vae=getattr(cfg, "vae_epochs", 20),
                epochs_gan=getattr(cfg, "gan_epochs", 100),
                batch=getattr(cfg, "batch_size", 256),
                privacy_cfg=privacy_cfg2,
                privacy_reg_cfg=privreg_off,
                mine_cfg={"enabled": False, "lambda": 0.0}
            )
            syn_noreg = hg_noreg.sample(len(df), shock=_parse_shock_from_cfg(cfg) if _parse_bool(getattr(cfg,"shock_enabled",False)) else ShockSpec())
            if pdcal is not None:
                syn_noreg["pd_12m"] = pdcal.predict_pd(syn_noreg)
                syn_noreg["pd_monthly"] = 1.0 - (1.0 - syn_noreg["pd_12m"].clip(0,1))**(1/12)
            if 'bad_within_horizon' not in syn_noreg.columns:
                if "pd_12m" in syn_noreg.columns:
                    rng = np.random.default_rng(11)
                    syn_noreg['bad_within_horizon'] = (rng.random(len(syn_noreg)) < syn_noreg["pd_12m"].values).astype(int)
                elif 'bad_within_horizon' in df.columns:
                    base = float(df['bad_within_horizon'].mean())
                    syn_noreg['bad_within_horizon'] = (np.random.rand(len(syn_noreg)) < base).astype(int)

            _RESERVED = {
                "bad_within_horizon", "default_flag",
                "pd_12m", "pd_monthly", "pd_12m_cal", "pd_monthly_cal", "pd", "pd_score"
            }
            cont_b = [c for c in cont_cols if (c in df.columns and c in syn_noreg.columns and c not in _RESERVED)]
            cat_b  = [c for c in cat_cols  if (c in df.columns and c in syn_noreg.columns and c not in _RESERVED)]
            fid_b  = univariate_fidelity(df, syn_noreg, cont_cols)
            corr_b = correlation_diff(df, syn_noreg, [c for c in cont_cols if c != "loan_balance"])
            util_b = generic_binary_downstream_eval(
                train_df=syn_noreg, test_df=df, target_col='bad_within_horizon',
                cont_cols=cont_b, cat_cols=cat_b
            )
            np.savez(os.path.join(bdir, "eval_scores_noreg.npz"),
                y=util_b["y_true"], p=util_b["scores"])
            mia_b  = membership_inference_auc(df, syn_noreg, cont_cols)

            bdir = os.path.join(base_dir, "no_privreg")
            os.makedirs(bdir, exist_ok=True)
            syn_noreg.to_csv(os.path.join(bdir, "originations_synth.csv"), index=False)
            fid_b.to_csv(os.path.join(bdir, "fidelity_univariate.csv"), index=False)
            with open(os.path.join(bdir, "summary.txt"), "w") as f:
                f.write(f"Ablation (no priv-regularizer)\n")
                f.write(f"Correlation avg abs diff: {corr_b['avg_abs_corr_diff']:.4f}\n")
                f.write(f"Utility PD AUC(synth->real): {util_b.get('auc')}\n")
                f.write(f"Utility PD Brier(synth->real): {util_b.get('brier')}\n")
                f.write(f"Membership inference AUC: {mia_b}\n")
                if hasattr(hg_noreg, 'train_eps_log'): f.write(f"DP eps log: {hg_noreg.train_eps_log}\n")

            try:
                evaluate_synth_vs_real(
                    df_real=df, df_synth=syn_noreg,
                    cont_cols=cont_cols, cat_cols=cat_cols,
                    default_col="bad_within_horizon",
                    pd_col=( "pd_12m" if "pd_12m" in df.columns else "pd_monthly"),
                    out_dir=os.path.join(bdir, "credit_eval"),
                    disperse_with_quantiles=getattr(cfg, "disperse_quantiles", False),
                )
            except Exception:
                pass
    except Exception as e:
        with open(f"{cfg.output_dir}/pipeline_warning.txt","a") as f:
            f.write("\n[baselines] " + str(e))

    # 7) Save + metrics-focused outputs (+ minimal visuals only if enabled)
    out = cfg.output_dir
    syn_o = normalize_df(
        syn_o, column_map_path=(colmap_path or None), alias_pack_path=(alias_path or None), parse_dates=True
    )
    syn_o = _ensure_pd_columns(syn_o)
    df.to_csv(f"{out}/originations_real.csv", index=False)
    syn_o.to_csv(f"{out}/originations_synth.csv", index=False)
    fid.to_csv(f"{out}/fidelity_univariate.csv", index=False)
    with open(f"{out}/summary.txt","w") as f:
        f.write(f"Correlation avg abs diff: {corr['avg_abs_corr_diff']:.4f}\n")
        f.write(f"Utility PD AUC(synth->real): {util.get('auc')}\n")
        f.write(f"Utility PD Brier(synth->real): {util.get('brier')}\n")
        f.write(f"Membership inference AUC: {mia}\n")
        f.write(f"DCR synth mean: {dcr.get('dcr_synth_mean')}\n")
        f.write(f"Reident rate @ real p1 tau: {dcr.get('reident_rate_tau_p1')}\n")
        if hasattr(hg, 'train_eps_log'):
            f.write(f"DP eps log (ε is approx RDP): {hg.train_eps_log}\n")
        if hasattr(hg, 'priv_reg'):     f.write(f"PrivReg cfg: {getattr(hg,'priv_reg',{})}\n")
        if hasattr(hg, 'train_logs'):   f.write(f"Train logs: {getattr(hg,'train_logs',{})}\n")

    # Minimal visuals (skip heavy plots if eval_minimal=True)
    try:
        if not eval_minimal:
            viz_top_k = getattr(cfg, "viz_top_k", None)
            if viz_top_k is None:
                viz_top_k = min(12, len(cont_cols))
            cols_to_plot = cont_cols[:viz_top_k]
            plot_histograms(df, syn_o, cols_to_plot, out)
            plot_pca(df, syn_o, [c for c in cont_cols if c != "loan_balance"], out)
    except Exception as e:
        with open(f"{out}/pipeline_warning.txt","a") as f: f.write("\n[viz] "+str(e))

    # Correlation compare + report (skip when minimal)
    try:
        if not eval_minimal:
            common_numeric = [c for c in df.columns if c in syn_o.columns and pd.api.types.is_numeric_dtype(df[c])]
            stats_corr = plot_corr_compare(df, syn_o, common_numeric, cfg.output_dir, method="pearson",
                                           fname_prefix="corr_compare")
            with open(f"{cfg.output_dir}/corr_compare_stats.txt", "w") as f:
                f.write(str(stats_corr))
    except Exception as e:
        with open(f"{out}/pipeline_warning.txt","a") as f: f.write("\n[corr_compare] "+str(e))

    try:
        build_report(out)
    except Exception as e:
        with open(f"{out}/pipeline_warning.txt","a") as f: f.write("\n[report] "+str(e))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv)>1 else "configs/universal_example_credit.yaml")
