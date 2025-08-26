import argparse, yaml
from pipelines.learn_from_csv import run as run_universal

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    mode = cfg.get("mode","learned")
    adapter = cfg.get("adapter","universal").lower()
    if mode=="learned" and adapter=="universal":
        run_universal(args.config)
    else:
        raise ValueError(f"Unsupported config: mode={mode}, adapter={adapter}")

if __name__=="__main__":
    main()
