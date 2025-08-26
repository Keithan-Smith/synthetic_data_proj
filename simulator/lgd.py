import numpy as np

def beta_params_from_mean_var(mean, var):
    m = np.clip(mean, 1e-3, 1-1e-3)
    v = min(var, m*(1-m)/3)
    k = m*(1-m)/v - 1
    alpha = max(0.5, m*k); beta = max(0.5, (1-m)*k)
    return alpha, beta

def collateral_score(property_code: str, other_debtors: str):
    s = 0.0
    if property_code in ("A121","A122"): s -= 0.07
    if other_debtors in ("A103","A102"): s -= 0.04
    return s

def lgd_mean(ltv, unemployment, property_code, other_debtors):
    return 0.45 + 0.25*ltv + 1.0*(unemployment-0.05) + collateral_score(property_code, other_debtors)

def sample_lgd(balance, loan_amount, unemployment, property_code, other_debtors, var=0.015, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    ltv = balance / max(loan_amount, 1e-6)
    m = lgd_mean(ltv, unemployment, property_code, other_debtors)
    a,b = beta_params_from_mean_var(m, var)
    return float(rng.beta(a,b))
