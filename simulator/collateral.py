import numpy as np
import pandas as pd
from typing import Dict, Optional

# Support both raw codes and mapped labels
GERMAN_PURPOSE_TO_ASSET = {
    "A40": "vehicle", "A41": "vehicle",
    "car_new": "vehicle", "car_used": "vehicle",
}

def _norm(x) -> str:
    try:
        return str(x).strip()
    except Exception:
        return ""

def infer_asset_type(
    row: pd.Series,
    purpose_col: str = "purpose",
    purpose_to_asset: Optional[Dict[str, str]] = None,
    mortgage_heuristic: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[str]:
    """
    1) If purpose matches mapping -> asset_type.
    2) Else, optional mortgage heuristic:
         if housing indicates ownership AND property indicates real estate AND term >= min_term -> 'mortgage' w.p. p.
    3) Else None (unsecured).
    """
    purpose_to_asset = purpose_to_asset or GERMAN_PURPOSE_TO_ASSET
    rng = rng or np.random.default_rng(12345)

    pv = _norm(row.get(purpose_col, ""))
    if pv in purpose_to_asset:
        return purpose_to_asset[pv]

    mh = mortgage_heuristic or {}
    if str(mh.get("enabled", False)).lower() in ("true", "1", "yes"):
        housing_val  = _norm(row.get(mh.get("housing_col", "housing"), ""))
        property_val = _norm(row.get(mh.get("property_col", "property"), ""))
        term_col     = mh.get("term_col", "loan_term")
        term_val     = row.get(term_col, None)

        owner_code = _norm(mh.get("owner_code", "A152"))      # "own"
        re_code    = _norm(mh.get("real_estate_code", "A121"))# "real_estate"
        min_term   = int(mh.get("min_term_months", 60))
        prob       = float(mh.get("p_mortgage_if_eligible", 0.35))

        # also support mapped labels if enum_map was applied
        owner_ok = housing_val in {owner_code, "own"}
        re_ok    = property_val in {re_code, "real_estate"}
        term_ok  = False
        if term_val is not None:
            try:
                term_ok = float(term_val) >= float(min_term)
            except Exception:
                term_ok = False

        if owner_ok and re_ok and term_ok:
            if rng.random() < prob:
                return "mortgage"

    return None

# -------- initial asset value models --------

def _beta_ltv_asset(loan_amount: float, cfg: dict, rng: np.random.Generator) -> float:
    a, b = (cfg.get("ltv_beta") or [8, 2])
    ltv0 = np.clip(rng.beta(a, b), 0.2, 0.99)
    return loan_amount / max(ltv0, 1e-6)

def _lognormal_asset(loan_amount: float, cfg: dict, rng: np.random.Generator) -> float:
    mu   = float(cfg.get("lognormal_mu", 11.0))
    sig  = float(cfg.get("lognormal_sigma", 0.25))
    val  = rng.lognormal(mean=mu, sigma=sig)
    if cfg.get("anchor_to_loan", False):
        tl = float(cfg.get("target_ltv_mean", 0.8))
        target = loan_amount / max(tl, 1e-6)
        val = target if val <= 0 else val * (target / val)
    return max(val, 1e-6)

def initial_asset_value(loan_amount: float, asset_type: Optional[str],
                        cfg_all: dict, rng: np.random.Generator) -> Optional[float]:
    if asset_type is None:
        return None
    cfg = (cfg_all or {}).get(asset_type, {})
    if str(cfg.get("init_model", "beta_ltv")).lower() == "lognormal_asset":
        return _lognormal_asset(loan_amount, cfg, rng)
    return _beta_ltv_asset(loan_amount, cfg, rng)

# -------- monthly evolution --------

def evolve_asset(prev_value: float, asset_type: Optional[str], macro_row: dict, cfg_all: dict,
                 rng: Optional[np.random.Generator] = None) -> Optional[float]:
    if asset_type is None or prev_value is None:
        return None
    rng = rng or np.random.default_rng(123)
    cfg = (cfg_all or {}).get(asset_type, {})
    model = str(cfg.get("evolve", "rule")).lower()

    if model == "gbm":
        mu = float(cfg.get("gbm_mu", 0.001))
        sigma = float(cfg.get("gbm_sigma", 0.01))
        z = rng.normal(0.0, 1.0)
        return max(0.0, prev_value * np.exp((mu - 0.5*sigma**2) + sigma*z))

    if model == "inflation_linked":
        hpi = float(cfg.get("hpi_sensitivity", 0.8))
        infl = float(macro_row.get("inflation_m", 0.0))
        noise = float(cfg.get("noise_sigma", 0.0)) * rng.normal(0, 1)
        return max(0.0, prev_value * (1.0 + hpi*infl + noise))

    # defaults by type
    if asset_type == "vehicle":
        dep = float(cfg.get("monthly_dep", 0.015))
        noise = float(cfg.get("noise_sigma", 0.0)) * rng.normal(0, 1)
        shock = float(macro_row.get("gdp_growth_m", 0.0)) * 0.1
        return max(0.0, prev_value * (1.0 - dep + shock + noise))
    elif asset_type == "mortgage":
        hpi = float(cfg.get("hpi_sensitivity", 0.8))
        infl = float(macro_row.get("inflation_m", 0.0))
        noise = float(cfg.get("noise_sigma", 0.0)) * rng.normal(0, 1)
        return max(0.0, prev_value * (1.0 + hpi*infl + noise))
    return prev_value

# -------- secured recovery & LGD --------

def secured_recovery(ead: float, asset_value: float, cfg: dict) -> float:
    haircut = float(cfg.get("haircut", 0.25))
    costs   = float(cfg.get("costs_pct", 0.08))
    proceeds = max(0.0, asset_value * (1.0 - haircut))
    return max(0.0, proceeds - ead * costs)
