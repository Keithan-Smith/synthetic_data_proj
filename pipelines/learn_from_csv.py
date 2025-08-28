"""
Lean orchestrator:
  1) Load data (CSV via path OR UCI via ucimlrepo id)
  2) Infer schema, pick cont/cat (exclude targets)
  3) Train HybridGenerator (Copula+VAE+GAN+Cat-AR)
  4) Sample synthetic data (optionally with ShockSpec from YAML)
  5) (optional) credit portfolio simulation
  6) Eval (fidelity, utility, privacy), visuals, HTML report
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif, chi2

from utils.config import load_config
from data.adapters.universal_csv import load_any_csv, normalize_df
from data.schemas import infer_schema_from_df
from hybrid.hybrid import HybridGenerator
from shocks.spec import ShockSpec

from eval.fidelity import univariate_fidelity, correlation_diff
from eval.privacy import membership_inference_auc
from eval.utility import generic_binary_downstream_eval
from eval.viz import plot_histograms, plot_corr, plot_pca
from eval.report import build_report


# -------- optional credit sim --------
def _maybe_credit_bits():
    try:
        from simulator.portfolio import simulate_portfolio
        from simulator.pd_calibration import PDCalibrator
        return simulate_portfolio, PDCalibrator
    except Exception:
        return None, None


# -------- helpers --------
def _resolve_with_cfg(base_dir, pathlike):
    if not pathlike:
        return ""
    if os.path.isabs(pathlike):
        return pathlike
    return os.path.join(base_dir or "", pathlike)


def _parse_shock_from_cfg(cfg) -> ShockSpec:
    """
    Accepts in YAML either list- or dict-form for corr_pairs.
    shock:
      corr_shrink: 0.25
      corr_pairs:
        - [income, loan_amount, 0.6]
        - [age, loan_amount, -0.2]
      cont_mu_shift: {income: 1000}
      cont_scale: {loan_amount: 1.1}
      residual_scale: 1.2
      cat_logit_bias:
        purpose:
          A40: 0.5
    or:
      shock:
        corr_pairs:
          "(income,loan_amount)": 0.6
    """
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
            # merge targets; coerce to canonical 'bad_within_horizon' if possible
            for col in y.columns:
                if col in df.columns:
                    df[f"target_{col}"] = y[col]
                else:
                    df[col] = y[col]
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


# ---------------- main pipeline ----------------
def run(cfg_path: str):
    cfg = load_config(cfg_path)
    # ensure config_dir exists for relative paths
    if not getattr(cfg, "config_dir", None):
        cfg.config_dir = os.path.dirname(os.path.abspath(cfg_path))

    os.makedirs(cfg.output_dir, exist_ok=True)

    # CUDA fallback if needed
    if str(getattr(cfg, "device", "cuda")).lower() == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU.")
        cfg.device = "cpu"

    # 1) load data
    df = _load_df(cfg)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded dataframe is empty. Check data_source/data_path/uci_id and mapping files.")

    # 2) schema + feature sets (exclude obvious target-like columns)
    schema = infer_schema_from_df(df.select_dtypes(exclude=["datetime", "datetimetz"]))

    PREFERRED_TARGETS = {
        "bad_within_horizon","default","default_flag","class","label","target",
        "mortality_30d","readmission_30d"
    }
    target_like = {c for c in df.columns if c in PREFERRED_TARGETS or df[c].dropna().nunique() == 2}

    # infer continuous/categorical; allow YAML overrides
    cont_cols = (getattr(cfg, "cont_cols", None) or schema.cont_cols())
    cat_cols  = (getattr(cfg, "cat_cols", None)  or schema.cat_cols())

    # keep only present + exclude target-like from modeling features
    cont_cols = [c for c in cont_cols if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]
    cat_cols  = [c for c in cat_cols if c in df.columns
                 and not pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]

    # robust diagnostics
    if len(cont_cols) == 0:
        numeric_present = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(
            "No continuous columns detected after filtering. "
            "Provide 'cont_cols' in YAML, or adjust mapping/aliases. "
            f"Numeric columns seen: {numeric_present[:20]} (showing up to 20). "
            f"Target-like excluded: {sorted(list(target_like))[:20]}"
        )

    # 3) train hybrid
    try:
        hg = HybridGenerator(cont_cols=cont_cols, cat_cols=cat_cols, schema=schema, device=cfg.device)
        privacy_cfg = {
            "enabled": getattr(cfg, "privacy_enabled", False),
            "dp_max_grad_norm": getattr(cfg, "dp_max_grad_norm", 1.0),
            "dp_noise_multiplier": getattr(cfg, "dp_noise_multiplier", 0.0),
            "dp_delta": getattr(cfg, "dp_delta", 1e-5),
        }
        hg.fit(df, epochs_vae=getattr(cfg, "vae_epochs", 20),
               epochs_gan=getattr(cfg, "gan_epochs", 100),
               batch=getattr(cfg, "batch_size", 256),
               privacy_cfg=privacy_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize/train HybridGenerator. "
                           f"cont_cols={cont_cols}, cat_cols={cat_cols}. "
                           f"Details: {e}") from e

    # === Fit a PDCalibrator on the real DF (for labeling synthetic + simulator) ===
    simulate_portfolio, PDCalibrator = _maybe_credit_bits()
    pdcal = None
    preferred_bin = ["bad_within_horizon","default","default_flag","class","label","target"]
    bin_targets = [c for c in df.columns
                   if not isinstance(df[c], pd.DataFrame) and df[c].dropna().nunique()==2]
    if bin_targets:
        tcol = next((c for c in preferred_bin if c in bin_targets), bin_targets[0])
        # ensure canonical target exists on REAL df
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

    # 4) sample synthetic originations (with or without shocks)
    use_shock = bool(getattr(cfg, "shock_enabled", True))
    shock_spec = _parse_shock_from_cfg(cfg) if use_shock else ShockSpec()  # neutral if disabled
    syn_o = hg.sample(len(df), shock=shock_spec)

    # Add synthetic binary target using PDCalibrator (or baseline rate fallback)
    if 'bad_within_horizon' not in syn_o.columns:
        if pdcal is not None:
            p_syn = pdcal.predict_pd(syn_o)
            syn_o['bad_within_horizon'] = (np.random.rand(len(p_syn)) < p_syn).astype(int)
        elif 'bad_within_horizon' in df.columns:
            base = float(df['bad_within_horizon'].mean())
            syn_o['bad_within_horizon'] = (np.random.rand(len(syn_o)) < base).astype(int)

    # 5) optional credit portfolio panel
    if getattr(cfg, "task", "none") == "credit_portfolio" and simulate_portfolio is not None:
        panel = simulate_portfolio(
            hybrid_model=hg, originations_df=df,
            months=getattr(cfg, "months", 18),
            start_date=getattr(cfg, "start_date", "2019-01-31"),
            init_customers=getattr(cfg, "init_customers", 100000),
            base_new=getattr(cfg, "base_new", 120),
            pd_calibrator=pdcal,
            pd_logit_shift=getattr(cfg, "pd_logit_shift", 0.0),
            macro_pd_mult=getattr(cfg, "macro_pd_mult", 1.0)
        )
        panel.to_csv(f"{cfg.output_dir}/panel_synth.csv", index=False)

    # 6) evaluation
    cont_eval = [c for c in cont_cols if c != "loan_balance"]
    fid  = univariate_fidelity(df, syn_o, cont_cols)
    corr = correlation_diff(df, syn_o, cont_eval)
    # pass explicit target (you patched eval.utility to accept this)
    util = generic_binary_downstream_eval(train_df=syn_o, test_df=df, target_col='bad_within_horizon')
    mia  = membership_inference_auc(df, syn_o, cont_cols)

    # 7) save + visuals + report
    out = cfg.output_dir
    df.to_csv(f"{out}/originations_real.csv", index=False)
    syn_o.to_csv(f"{out}/originations_synth.csv", index=False)
    fid.to_csv(f"{out}/fidelity_univariate.csv", index=False)
    with open(f"{out}/summary.txt","w") as f:
        f.write(f"Correlation avg abs diff: {corr['avg_abs_corr_diff']:.4f}\n")
        f.write(f"Utility PD AUC(synth->real): {util.get('auc')}\n")
        f.write(f"Utility PD Brier(synth->real): {util.get('brier')}\n")
        f.write(f"Membership inference AUC: {mia}\n")
        if hasattr(hg, 'train_eps_log'): f.write(f"DP eps log: {hg.train_eps_log}\n")

    try:
        viz_top_k = getattr(cfg, "viz_top_k", None)
        cols_to_plot = cont_cols if viz_top_k is None else cont_cols[:viz_top_k]
        plot_histograms(df, syn_o, cols_to_plot, out)
        plot_corr(df, syn_o, cont_eval[:8], out)
        plot_pca(df, syn_o, cont_eval, out)
        build_report(out)
    except Exception as e:
        with open(f"{out}/pipeline_warning.txt","w") as f: f.write(str(e))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv)>1 else "configs/universal_example_credit.yaml")
