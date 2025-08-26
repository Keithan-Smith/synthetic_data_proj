import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def simulate_macro(start_date="2019-01-31", months=12,
                   gdp_annual_mean=0.02, infl_annual_mean=0.03, unemp_mean=0.05, seed=11):
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp(start_date) + relativedelta(months=i) for i in range(months)]
    gdp_m_mean = (1 + gdp_annual_mean) ** (1/12) - 1
    infl_m_mean = (1 + infl_annual_mean) ** (1/12) - 1
    phi = 0.6
    shocks = rng.normal(scale=[0.0015, 0.002, 0.002], size=(months, 3))
    gdp = np.zeros(months); infl = np.zeros(months); unemp = np.zeros(months)
    gdp[0] = gdp_m_mean + shocks[0,0]
    infl[0] = infl_m_mean + shocks[0,1]
    unemp[0] = unemp_mean + shocks[0,2]
    for t in range(1, months):
        gdp[t] = gdp_m_mean + phi*(gdp[t-1]-gdp_m_mean) + shocks[t,0]
        infl[t] = infl_m_mean + phi*(infl[t-1]-infl_m_mean) + shocks[t,1]
        unemp[t] = np.clip(unemp_mean + phi*(unemp[t-1]-unemp_mean) + shocks[t,2], 0.02, 0.2)
    return pd.DataFrame({"snapshot_date": dates,
                         "gdp_growth_m": gdp, "inflation_m": infl, "unemployment": unemp})
