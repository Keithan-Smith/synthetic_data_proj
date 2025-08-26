import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PD_FEATURES_CONT = ["age","loan_amount","loan_term","installment_rate_pct","existing_credits","residence_since","number_of_dependents","income"]
PD_FEATURES_CAT  = ["checking_status","credit_history","purpose","savings","employment_since",
                    "other_debtors","property","other_installment_plans","job","telephone","foreign_worker",
                    "personal_status_sex","residential_status"]

class PDCalibrator:
    def __init__(self):
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.clf = LogisticRegression(max_iter=200)
        self.cat_cols = PD_FEATURES_CAT
        self.cont_cols = PD_FEATURES_CONT

    def _X(self, df):
        Xc = df[self.cont_cols].to_numpy()
        Xoh = self.enc.transform(df[self.cat_cols])
        return np.hstack([Xc, Xoh])

    def fit(self, df):
        have_cont = [c for c in self.cont_cols if c in df.columns]
        have_cat  = [c for c in self.cat_cols if c in df.columns]
        self.cont_cols = have_cont
        self.cat_cols  = have_cat
        self.enc.fit(df[self.cat_cols])
        X = self._X(df)
        y = df["bad_within_horizon"].to_numpy()
        self.clf.fit(X, y)
        return self

    def monthly_hazard(self, df):
        P = self.clf.predict_proba(self._X(df))[:,1]
        H = np.clip(df["loan_term"].to_numpy(), 1, 360) if "loan_term" in df.columns else 36
        h = 1 - (1 - P) ** (1/H)
        return np.clip(h, 5e-4, 0.2)
