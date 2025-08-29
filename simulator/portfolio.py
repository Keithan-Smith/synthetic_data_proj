import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict

from simulator.collateral import (
    infer_asset_type, initial_asset_value, evolve_asset, secured_recovery,
)

DEFAULT_PURPOSE_TO_ASSET = {"A40": "vehicle", "A41": "vehicle", "car_new": "vehicle", "car_used": "vehicle"}

def _month_range(start_date: str, months: int):
    d = datetime.fromisoformat(start_date)
    for k in range(months):
        yield (d + relativedelta(months=+k)).date().isoformat()

def _macro_row(gdp_base: float, infl_base: float, unemp_base: float, k: int) -> dict:
    rng = np.random.default_rng(1234 + k)
    return {
        "gdp_growth_m": gdp_base + rng.normal(0.0, 0.001),
        "inflation_m":  infl_base + rng.normal(0.0, 0.0015),
        "unemployment": np.clip(unemp_base + rng.normal(0.0, 0.001), 0.01, 0.25),
    }

# ---------- NEW: robust numeric getters ----------
def _first_present(df: pd.DataFrame, names) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return df[n]
    return None

def _numeric_series_like(df: pd.DataFrame, candidates, fill_value=0.0) -> pd.Series:
    """
    Return a float Series for any of the candidate columns if present; otherwise a
    length-matched float Series filled with fill_value.
    """
    s = _first_present(df, candidates)
    if s is None:
        return pd.Series(np.full(len(df), fill_value, dtype=float), index=df.index)
    return pd.to_numeric(s, errors="coerce").fillna(fill_value).astype(float)
# -------------------------------------------------

def _init_collateral(df: pd.DataFrame, collateral_cfg: dict,
                     purpose_to_asset: Dict[str,str], mortgage_heuristic: dict,
                     rng: np.random.Generator):
    ats, avs, l0s, lts = [], [], [], []
    for _, r in df.iterrows():
        at = infer_asset_type(r, purpose_col="purpose",
                              purpose_to_asset=purpose_to_asset,
                              mortgage_heuristic=mortgage_heuristic,
                              rng=rng)
        ats.append(at)
        if at is None:
            avs.append(None); l0s.append(None); lts.append(None)
        else:
            la = float(r.get("loan_amount", 0.0))
            av = initial_asset_value(la, at, collateral_cfg, rng)
            avs.append(av)
            l0 = (la / av) if (av and av > 0) else None
            l0s.append(l0); lts.append(l0)
    return ats, avs, l0s, lts

def simulate_portfolio(
    hybrid_model,
    originations_df: pd.DataFrame,
    months: int,
    start_date: str,
    init_customers: int,
    base_new: int,
    pd_calibrator=None,
    pd_logit_shift: float = 0.0,
    macro_pd_mult: float = 8.0,
    lgd_cfg: Optional[dict] = None,
    collateral_cfg: Optional[dict] = None,
    purpose_to_asset: Optional[Dict[str,str]] = None,
    mortgage_heuristic: Optional[dict] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    lgd_cfg = lgd_cfg or {
        "unsecured_beta": [2.0, 5.0],
        "vehicle":   {"ltv_beta": [8, 2],   "haircut": 0.30, "costs_pct": 0.10, "monthly_dep": 0.015},
        "mortgage":  {"ltv_beta": [8, 1.5], "haircut": 0.20, "costs_pct": 0.08, "hpi_sensitivity": 0.8},
    }
    collateral_cfg = collateral_cfg or lgd_cfg
    purpose_to_asset = purpose_to_asset or DEFAULT_PURPOSE_TO_ASSET
    mortgage_heuristic = mortgage_heuristic or {"enabled": False}

    # seed book (sample then ensure IDs & balances)
    book = originations_df.sample(n=min(init_customers, len(originations_df)), replace=True, random_state=7).copy()
    if "customer_id" not in book.columns:
        book["customer_id"] = [f"CUST_{i:06d}" for i in range(len(book))]

    # ---------- CHANGED: robust loan_balance init ----------
    # fallback order includes common aliases & raw German column name 'Attribute5'
    book["loan_balance"] = _numeric_series_like(
        book, ["loan_balance", "loan_amount", "credit_amount", "amount", "Attribute5"], fill_value=0.0
    )
    # -------------------------------------------------------

    book["customer_tenure"] = 0
    at, av, l0, lt = _init_collateral(book, collateral_cfg, purpose_to_asset, mortgage_heuristic, rng)
    book["asset_type"] = at
    book["asset_value"] = av
    book["ltv_init"] = l0
    book["ltv"] = lt

    rows = []
    gdp0, infl0, unemp0 = 0.0017, 0.0052, 0.052

    for t, snap in enumerate(_month_range(start_date, months)):
        macro = _macro_row(gdp0, infl0, unemp0, t)

        # new originations
        new_df = hybrid_model.sample(base_new)
        if "customer_id" not in new_df.columns:
            base_id = len(book) + len(new_df)
            new_df["customer_id"] = [f"CUST_{base_id+i:06d}" for i in range(len(new_df))]

        # ---------- CHANGED: robust for new_df as well ----------
        new_df["loan_balance"] = _numeric_series_like(
            new_df, ["loan_balance", "loan_amount", "credit_amount", "amount", "Attribute5"], fill_value=0.0
        )
        # --------------------------------------------------------

        new_df["customer_tenure"] = 0
        at, av, l0, lt = _init_collateral(new_df, collateral_cfg, purpose_to_asset, mortgage_heuristic, rng)
        new_df["asset_type"] = at
        new_df["asset_value"] = av
        new_df["ltv_init"] = l0
        new_df["ltv"] = lt

        # grow book
        book = pd.concat([book, new_df], ignore_index=True)

        # PD for everyone (unchanged)
        if pd_calibrator is not None:
            pdm = pd_calibrator.monthly_hazard(book)
            from scipy.special import expit, logit
            adj = expit(np.clip(logit(np.clip(pdm, 1e-6, 1-1e-6)) + pd_logit_shift
                                + macro_pd_mult * macro["unemployment"], -10, 10))
            pd_monthly = np.clip(adj, 1e-6, 0.5)
            pd_12m = 1.0 - np.power(1.0 - pd_monthly, 12.0)
        else:
            pd_monthly = np.full(len(book), 0.02)
            pd_12m = 1.0 - np.power(0.98, 12.0)

        u = np.random.default_rng(1000 + t).random(len(book))
        default_flag = (u < pd_monthly).astype(int)
        ead = np.where(default_flag == 1, book["loan_balance"].astype(float), np.nan)

        asset_vals_next = []
        recovery = np.zeros(len(book))
        loss = np.zeros(len(book))
        lgd = np.full(len(book), np.nan)

        for i, (df_, at_i, ead_i, av_prev) in enumerate(zip(default_flag, book["asset_type"], ead, book["asset_value"])):
            av_now = evolve_asset(av_prev, at_i, macro, collateral_cfg, rng) if av_prev is not None else None
            asset_vals_next.append(av_now)
            if df_ == 1:
                if at_i is None:
                    a, b = lgd_cfg.get("unsecured_beta", [2.0, 5.0])
                    lgd_i = np.random.default_rng(77 + t + i).beta(a, b)
                    rec = float(ead_i) * (1.0 - lgd_i)
                else:
                    rec = secured_recovery(float(ead_i), float(av_now or 0.0), collateral_cfg.get(at_i, {}))
                    rec = np.clip(rec, 0.0, float(ead_i))
                    lgd_i = 1.0 - (rec / float(ead_i) if ead_i and ead_i > 0 else 0.0)
                recovery[i] = rec
                loss[i] = float(ead_i) - rec
                lgd[i] = np.clip(lgd_i, 0.0, 1.0)

        out = book.copy()
        out["snapshot_date"] = snap
        out["pd_monthly"] = pd_monthly
        out["pd_12m"] = pd_12m
        out["default_flag"] = default_flag
        out["ead"] = ead
        out["recovery"] = recovery
        out["loss"] = loss
        out["lgd"] = lgd
        out["gdp_growth_m"] = macro["gdp_growth_m"]
        out["inflation_m"] = macro["inflation_m"]
        out["unemployment"] = macro["unemployment"]
        rows.append(out)

        # ---------- CHANGED: robust payment base ----------
        # Use 1% of "loan_amount" if present, else 1% of current "loan_balance"
        pay_base = _numeric_series_like(out, ["loan_amount", "credit_amount", "amount", "Attribute5"], fill_value=0.0)
        pay_base = np.where(pay_base > 0, pay_base, out["loan_balance"].to_numpy())
        pay = 0.01 * pay_base
        # ---------------------------------------------------

        surv = default_flag == 0
        new_bal = np.maximum(0.0, book["loan_balance"].to_numpy() - pay)
        new_bal = np.where(surv, new_bal, 0.0)
        book["loan_balance"] = new_bal
        book["customer_tenure"] = book["customer_tenure"].astype(int) + 1
        book["asset_value"] = asset_vals_next
        book["ltv"] = [(bal / av if (av and av > 0) else None) for bal, av in zip(book["loan_balance"], book["asset_value"])]
        book = book.loc[surv].reset_index(drop=True)

    return pd.concat(rows, ignore_index=True)
