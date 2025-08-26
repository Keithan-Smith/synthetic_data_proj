import numpy as np
import pandas as pd
from math import exp
from .macro import simulate_macro
from .pd_calibration import PDCalibrator
from .lgd import sample_lgd

def monthly_payment(principal, annual_rate, term_months):
    r = annual_rate/12.0
    if term_months <= 0: return principal
    if abs(r) < 1e-9: return principal / term_months
    return principal * r / (1 - (1 + r)**(-term_months))

def next_balance(balance, annual_rate, payment):
    r = annual_rate/12.0
    interest = balance * r
    principal_paid = max(0.0, payment - interest)
    return max(0.0, balance - principal_paid)

def simulate_portfolio(hybrid_model, originations_df, months=12, start_date="2019-01-31",
                       init_customers=500, base_new=120, seed=11, pd_calibrator=None,
                       pd_logit_shift=0.0, macro_pd_mult=8.0):
    rng = np.random.default_rng(seed)
    macro = simulate_macro(start_date=start_date, months=months, seed=seed)
    states = {}
    rows = []

    init_idx = np.arange(min(init_customers, len(originations_df)))
    for i in init_idx:
        r = originations_df.iloc[i]
        cid = r.get("customer_id", f"CUST_{i:06d}")
        payment = monthly_payment(r.get("loan_amount", 10000.0), r.get("interest_rate", 0.1), int(r.get("loan_term", 36)))
        states[cid] = {
            **r.to_dict(),
            "customer_id": cid,
            "status": "active",
            "treatment_stage": None,
            "months_since_default": None,
            "monthly_payment": payment
        }

    for t in range(months):
        mrow = macro.iloc[t]
        date = mrow["snapshot_date"].date()
        lam = base_new * exp(-8*(mrow["unemployment"]-0.05))
        n_new = max(0, int(rng.poisson(lam)))

        if hasattr(hybrid_model, "sample"):
            new = hybrid_model.sample(n_new)
        else:
            new = originations_df.sample(n_new, replace=True, random_state=seed+t)

        for j in range(len(new)):
            r = new.iloc[j]
            cid = f"NEW{t:02d}_{j:05d}"
            payment = monthly_payment(r.get("loan_amount", 10000.0), r.get("interest_rate", 0.1), int(r.get("loan_term", 36)))
            states[cid] = {**r.to_dict(),
                           "customer_id": cid, "loan_origination_date": date,
                           "status":"active","treatment_stage":None,"months_since_default":None,
                           "monthly_payment": payment, "customer_tenure": 0, "late_payments":0, "past_defaults":0}

        for cid, st in list(states.items()):
            if st["status"] == "exited": continue
            st["customer_tenure"] = int(st.get("customer_tenure",0)) + 1 if t>0 else st.get("customer_tenure",0)
            st["age"] = float(st.get("age",40)) + (1/12 if t>0 else 0)
            growth_nominal = 0.6*mrow["gdp_growth_m"] + 0.5*mrow["inflation_m"] + rng.normal(0,0.01)
            st["income"] = float(st.get("income",40000)) * (1 + growth_nominal) if t>0 else float(st.get("income",40000))

            default_flag = 0; lgd = np.nan
            if st["status"] == "active":
                if pd_calibrator is not None:
                    import pandas as _pd
                    base_pd = float(pd_calibrator.monthly_hazard(_pd.DataFrame([st]))[0])
                else:
                    base_pd = 0.006
                def _sigmoid(a):
                    import math
                    return 1.0/(1.0+math.exp(-a))
                import math
                base_pd = max(1e-6, min(0.5, base_pd))
                logit = math.log(base_pd/(1-base_pd)) + float(pd_logit_shift)
                p = _sigmoid(logit)
                p *= (1 + float(macro_pd_mult)*(mrow["unemployment"]-0.05))
                p *= (1 + 0.1*st.get("late_payments",0))
                p = np.clip(p, 5e-4, 0.2)
                if rng.random() < p:
                    default_flag = 1
                    st["status"] = "defaulted"
                    st["treatment_stage"] = rng.choice(["cure","exit","in_workout"], p=[0.45,0.25,0.30])
                    st["months_since_default"] = 0
                    lgd = sample_lgd(st.get("loan_balance", 10000.0), st.get("loan_amount", 10000.0), mrow["unemployment"],
                                     st.get("property","A124"), st.get("other_debtors","A101"), rng=rng)
                else:
                    st["loan_balance"] = next_balance(st.get("loan_balance", 10000.0), st.get("interest_rate", 0.1), st["monthly_payment"])
                    p_late = min(0.35, 0.02 + 0.5*p)
                    if rng.random() < p_late: st["late_payments"] = int(st.get("late_payments",0))+1
            elif st["status"] == "defaulted":
                st["months_since_default"] = (st["months_since_default"] or 0) + 1
                if st["treatment_stage"] == "cure" and st["months_since_default"] >= 3:
                    st["status"] = "active"
                    st["treatment_stage"] = None
                    st["months_since_default"] = None
                    st["interest_rate"] = min(0.35, st.get("interest_rate",0.1) + 0.0025)  # +25 bps
                    remaining_term = max(6, int(st.get("loan_term", 36)) - st["customer_tenure"])
                    st["monthly_payment"] = monthly_payment(st.get("loan_balance",10000.0), st["interest_rate"], remaining_term)
                    st["past_defaults"] = int(st.get("past_defaults",0)) + 1

            rows.append({
                "snapshot_date": date,
                **{k: st.get(k, None) for k in [
                    "customer_id","name","customer_tenure","loan_origination_date","age","income",
                    "marital_status","residential_status","number_of_dependents","loan_term","loan_amount",
                    "loan_balance","interest_rate","late_payments","past_defaults",
                    "installment_rate_pct","existing_credits","residence_since","checking_status","credit_history",
                    "purpose","savings","employment_since","other_debtors","property","other_installment_plans",
                    "job","telephone","foreign_worker"
                ]},
                "default_flag": default_flag,
                "lgd": lgd,
                "gdp_growth_m": mrow["gdp_growth_m"],
                "inflation_m": mrow["inflation_m"],
                "unemployment": mrow["unemployment"]
            })

        states = {k:v for k,v in states.items() if v["status"] != "exited"}
    return pd.DataFrame(rows)
