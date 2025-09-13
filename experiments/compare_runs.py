from __future__ import annotations
import os
import argparse
import json
import pandas as pd
from eval.delong import delong_two_model_test

def _load_scores(run_dir: str) -> pd.DataFrame:
    p = os.path.join(run_dir, "credit_eval", "gbm_scores_real.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Scores not found: {p}")
    return pd.read_csv(p)

def _load_metrics(run_dir: str):
    p = os.path.join(run_dir, "credit_eval", "gbm_metrics.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--label_a", default="Run A")
    ap.add_argument("--label_b", default="Run B")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    A = _load_scores(args.run_a)
    B = _load_scores(args.run_b)
    if len(A) != len(B) or (A["y"].astype(int).values != B["y"].astype(int).values).any():
        raise ValueError("Mismatched or differently ordered labels; ensure both runs used the same real test set.")

    res = delong_two_model_test(A["y"].astype(int).values, A["score"].values, B["score"].values)
    print(f"\n=== DeLong comparison ===\n{args.label_a} vs {args.label_b}")
    for k, v in res.items():
        print(f"{k}: {v}")

    if args.out_csv:
        pd.DataFrame([{
            "label_a": args.label_a, "label_b": args.label_b, **res
        }]).to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")

    # (Optional) also print the single-run GBM AUCs as a cross-check
    ma = _load_metrics(args.run_a).get("AUC_real")
    mb = _load_metrics(args.run_b).get("AUC_real")
    if ma is not None or mb is not None:
        print(f"\nGBM AUC (from JSON): {args.label_a}={ma}, {args.label_b}={mb}")

if __name__ == "__main__":
    main()
