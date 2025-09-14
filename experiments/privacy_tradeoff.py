import os, sys, re, json, subprocess
import yaml
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))

def _run_pipeline(temp_yaml_path):
    cmd = [sys.executable, os.path.join(ROOT, "pipelines", "learn_from_csv.py"), temp_yaml_path]
    subprocess.run(cmd, check=True)

def _write_overridden_yaml(base_yaml, overrides, tag):
    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # apply overrides
    cfg.update(overrides)

    # tag the output dir to keep runs separate
    base_out = cfg.get("output_dir", "outputs/credit_run")
    cfg["output_dir"] = os.path.join(base_out, tag)

    os.makedirs(os.path.dirname(base_yaml), exist_ok=True)
    tmp_path = os.path.join(os.path.dirname(base_yaml), f"_tmp_{tag}.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return tmp_path, cfg["output_dir"]

def _read_metrics(out_dir):
    out = {"tag": os.path.basename(out_dir)}

    # summary/metrics from pipeline
    mj = os.path.join(out_dir, "metrics.json")
    if os.path.exists(mj):
        with open(mj, "r") as f:
            d = json.load(f)
        out["utility_auc"] = d.get("utility_auc")
        out["utility_brier"] = d.get("utility_brier")
        out["mia_auc"] = d.get("mia_auc")
        dp = d.get("dp_eps", {}) or {}
        # record both eps if present; pick a simple aggregate too
        out["eps_vae"] = dp.get("vae_eps")
        out["eps_gan"] = dp.get("gan_eps")
        out["eps_max"] = max([v for v in [out["eps_vae"], out["eps_gan"]] if v is not None], default=None)

    # AR/AUC from CAP metrics JSON (written by eval_credit)
    cap_real = os.path.join(out_dir, "credit_eval", "cap_real_metrics.json")
    if os.path.exists(cap_real):
        with open(cap_real, "r") as f:
            d = json.load(f)
        out["ar_real"] = d.get("AR")
        out["auc_real"] = d.get("AUC")

    cap_synth = os.path.join(out_dir, "credit_eval", "cap_synth_metrics.json")
    if os.path.exists(cap_synth):
        with open(cap_synth, "r") as f:
            d = json.load(f)
        out["ar_synth"] = d.get("AR")
        out["auc_synth"] = d.get("AUC")

    return out

def _plot_tradeoff(df, xcol, xlabel, out_png):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # Utility curves
    if "utility_auc" in df:
        ax1.plot(df[xcol], df["utility_auc"], marker="o", label="Utility AUC")
    if "ar_real" in df and df["ar_real"].notna().any():
        ax1.plot(df[xcol], df["ar_real"], marker="s", label="AR (real PD)")

    # Privacy risk (lower is better)
    if "mia_auc" in df:
        ax2.plot(df[xcol], df["mia_auc"], marker="^", linestyle="--", label="MIA AUC", color="tab:red")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Utility (AUC / AR)")
    ax2.set_ylabel("Privacy risk (MIA AUC ↓)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)

def sweep_dp(base_yaml, mus=(0.0, 0.4, 0.6, 0.8, 1.0)):
    rows = []
    for mu in mus:
        tag = f"dp_mu_{mu:.2f}"
        overrides = {
            "privacy_enabled": True,
            "dp_noise_multiplier": float(mu),
            # keep MMD fixed or off for clean attribution:
            "privreg_enabled": False,
            "shock_enabled": False,
        }
        tmp_yaml, out_dir = _write_overridden_yaml(base_yaml, overrides, tag)
        _run_pipeline(tmp_yaml)
        m = _read_metrics(out_dir)
        # convenient x-values
        m["mu"] = mu
        m["x_eps"] = m.get("eps_max")
        rows.append(m)
    return pd.DataFrame(rows)

def sweep_mmd(base_yaml, lambdas=(0.0, 0.25, 0.5, 1.0, 2.0), mode="repulsive"):
    rows = []
    for lam in lambdas:
        tag = f"mmd_{mode}_{lam:g}"
        overrides = {
            "privacy_enabled": False,             # isolate MMD effect
            "privreg_enabled": True,
            "privreg_mode": mode,                 # 'repulsive' (privacy-leaning) or 'attractive' (utility-leaning)
            "privreg_lambda_gan": float(lam),
            "privreg_lambda_vae": float(lam) * 0.25,
            "shock_enabled": False,
        }
        tmp_yaml, out_dir = _write_overridden_yaml(base_yaml, overrides, tag)
        _run_pipeline(tmp_yaml)
        m = _read_metrics(out_dir)
        m["lambda"] = lam
        m["x_lambda"] = lam
        rows.append(m)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    base_yaml = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "configs", "universal_example_credit.yaml")
    out_plots = os.path.join(ROOT, "outputs", "tradeoff_plots")
    os.makedirs(out_plots, exist_ok=True)

    # --- DP sweep (ε / μ vs AUC/AR & MIA) ---
    df_dp = sweep_dp(base_yaml)
    df_dp.to_csv(os.path.join(out_plots, "dp_sweep.csv"), index=False)
    # prefer plotting against epsilon if available; otherwise against mu
    if df_dp["x_eps"].notna().any():
        _plot_tradeoff(df_dp.sort_values("x_eps"), "x_eps", "ε (approx RDP)", os.path.join(out_plots, "dp_tradeoff_epsilon.png"))
    _plot_tradeoff(df_dp.sort_values("mu"), "mu", "Noise multiplier μ", os.path.join(out_plots, "dp_tradeoff_mu.png"))

    # --- MMD sweep (λ vs AUC/AR & MIA) ---
    df_mmd = sweep_mmd(base_yaml)
    df_mmd.to_csv(os.path.join(out_plots, "mmd_sweep.csv"), index=False)
    _plot_tradeoff(df_mmd.sort_values("x_lambda"), "x_lambda", "MMD weight λ", os.path.join(out_plots, "mmd_tradeoff_lambda.png"))
