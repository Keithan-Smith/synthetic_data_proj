# synth_project/pipelines/learn_from_csv.py
"""
Lean orchestrator:
  1) Load data (CSV via path OR UCI via ucimlrepo id)
  2) Infer schema, pick cont/cat (exclude targets)
  3) Train HybridGenerator (Copula+VAE+GAN+Cat-AR)
  4) Sample synthetic data
  5) (optional) credit portfolio simulation
  6) Eval (fidelity, utility, privacy), visuals, HTML report
"""

import os
import pandas as pd

from utils.config import load_config
from data.adapters.universal_csv import load_any_csv, normalize_df
from data.schemas import infer_schema_from_df
from hybrid.hybrid import HybridGenerator

from eval.fidelity import univariate_fidelity, correlation_diff
from eval.privacy import membership_inference_auc
from eval.utility import generic_binary_downstream_eval
from eval.viz import plot_histograms, plot_corr, plot_pca
from eval.report import build_report


# -------- optional credit sim (loaded only if used) --------
def _maybe_credit_bits():
    try:
        from simulator.portfolio import simulate_portfolio
        from simulator.pd_calibration import PDCalibrator
        return simulate_portfolio, PDCalibrator
    except Exception:
        return None, None


# ---------------- data loading ----------------
def _load_df(cfg) -> pd.DataFrame:
    """
    If cfg.data_source == 'ucimlrepo' and cfg.uci_id is set → fetch X & y and concat.
    Else → CSV/URL via load_any_csv.
    Then apply optional normalize_df (mapping/aliases/date parsing).
    """
    if getattr(cfg, "data_source", "").lower() == "ucimlrepo":
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
            # merge all target columns; normalize common German label to 0/1 if seen
            for col in y.columns:
                if col in df.columns:
                    df[f"target_{col}"] = y[col]
                else:
                    df[col] = y[col]
            # try normalize first target to bad_within_horizon
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

        # apply optional mapping/alias pack
        return normalize_df(
            df,
            column_map_path=(getattr(cfg, "column_map", "") or None),
            alias_pack_path=(getattr(cfg, "domain_pack", "") or None),
            parse_dates=True,
        )

    # CSV / URL path
    return load_any_csv(
        getattr(cfg, "data_path", ""),
        column_map_path=(getattr(cfg, "column_map", "") or None),
        alias_pack_path=(getattr(cfg, "domain_pack", "") or None),
        parse_dates=True,
        read_csv_kwargs=(getattr(cfg, "read_csv_kwargs", None) or None),
    )


# ---------------- main pipeline ----------------
def run(cfg_path: str):
    cfg = load_config(cfg_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) load data
    df = _load_df(cfg)

    # 2) schema + feature sets (exclude target-like columns)
    schema = infer_schema_from_df(df.select_dtypes(exclude=["datetime", "datetimetz"]))

    PREFERRED_TARGETS = {
        "bad_within_horizon","default","default_flag","class","label","target",
        "mortality_30d","readmission_30d"
    }
    target_like = {c for c in df.columns if c in PREFERRED_TARGETS or df[c].dropna().nunique() == 2}

    cont_cols = (cfg.cont_cols or schema.cont_cols())
    cont_cols = [c for c in cont_cols if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]

    cat_cols = (cfg.cat_cols or schema.cat_cols())
    cat_cols = [c for c in cat_cols if c in df.columns
                and not pd.api.types.is_numeric_dtype(df[c]) and c not in target_like]

    # 3) train hybrid
    hg = HybridGenerator(cont_cols=cont_cols, cat_cols=cat_cols, schema=schema, device=cfg.device)
    privacy_cfg = {
        "enabled": getattr(cfg, "privacy_enabled", False),
        "dp_max_grad_norm": getattr(cfg, "dp_max_grad_norm", 1.0),
        "dp_noise_multiplier": getattr(cfg, "dp_noise_multiplier", 0.0),
        "dp_delta": getattr(cfg, "dp_delta", 1e-5),
    }
    hg.fit(df, epochs_vae=cfg.vae_epochs, epochs_gan=cfg.gan_epochs, batch=cfg.batch_size, privacy_cfg=privacy_cfg)

    # 4) sample synthetic originations
    syn_o = hg.sample(len(df))

    # 5) optional credit portfolio panel
    if getattr(cfg, "task", "none") == "credit_portfolio":
        simulate_portfolio, PDCalibrator = _maybe_credit_bits()
        pdcal = None
        if PDCalibrator is not None:
            # fit a PD calibrator if a binary target exists
            bin_targets = [c for c in df.columns if df[c].dropna().nunique() == 2]
            if bin_targets:
                pdcal = PDCalibrator()
                tmp = df.copy()
                keep = [c for c in (pdcal.cont_cols + pdcal.cat_cols) if c in tmp.columns]
                if keep:
                    tmp = tmp[keep + [bin_targets[0]]].rename(columns={bin_targets[0]: "bad_within_horizon"})
                    pdcal.fit(tmp)
        if simulate_portfolio is not None:
            panel = simulate_portfolio(
                hybrid_model=hg, originations_df=df,
                months=cfg.months, start_date=cfg.start_date,
                init_customers=cfg.init_customers, base_new=cfg.base_new,
                pd_calibrator=pdcal, pd_logit_shift=cfg.pd_logit_shift, macro_pd_mult=cfg.macro_pd_mult
            )
            panel.to_csv(f"{cfg.output_dir}/panel_synth.csv", index=False)

    # 6) evaluation
    fid  = univariate_fidelity(df, syn_o, cont_cols)
    corr = correlation_diff(df, syn_o, [c for c in cont_cols if c != "loan_balance"])
    util = generic_binary_downstream_eval(train_df=syn_o, test_df=df)
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
        plot_histograms(df, syn_o, cont_cols[:6], out)
        plot_corr(df, syn_o, [c for c in cont_cols if c!='loan_balance'][:8], out)
        plot_pca(df, syn_o, [c for c in cont_cols if c!='loan_balance'], out)
        build_report(out)
    except Exception as e:
        with open(f"{out}/pipeline_warning.txt","w") as f: f.write(str(e))


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else "configs/universal_example_credit.yaml")
