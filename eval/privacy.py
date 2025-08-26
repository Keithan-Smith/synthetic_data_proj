import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

def membership_inference_auc(train_df: pd.DataFrame, synth_df: pd.DataFrame, cont_cols):
    real = train_df[cont_cols].copy(); real["is_real"] = 1
    syn  = synth_df[cont_cols].copy(); syn["is_real"] = 0
    data = pd.concat([real, syn], axis=0).dropna()
    y = data["is_real"].to_numpy(); X = data.drop(columns=["is_real"]).to_numpy()
    if len(set(y))<2: return float('nan')
    X, y = shuffle(X, y, random_state=7)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    scores = clf.predict_proba(X)[:,1]
    return float(roc_auc_score(y, scores))
