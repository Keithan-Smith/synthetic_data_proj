from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class RunConfig:
    mode: str = "learned"
    adapter: str = "universal"
    data_path: str = ""
    column_map: str = ""          # optional mapping file
    domain_pack: str = ""         # optional alias pack (credit/health/custom)
    output_dir: str = "reports"
    seed: int = 42
    device: str = "cuda"

    # pluggable downstream task: "none" | "credit_portfolio"
    task: str = "none"

    months: int = 12
    start_date: str = "2019-01-31"
    init_customers: int = 500
    base_new: int = 120

    training_profile: str = "balanced"
    cont_cols: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)
    vae_epochs: int = 0
    gan_epochs: int = 0
    batch_size: int = 0

    privacy_enabled: bool = False
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 0.0
    dp_delta: float = 1e-5
    mine_enabled: bool = False
    mine_lambda: float = 0.0

    pd_logit_shift: float = 0.0
    macro_pd_mult: float = 8.0

    shock: Dict[str, Any] = field(default_factory=dict)
    eval_enabled: bool = True

def _apply_training_profile(rc: RunConfig) -> RunConfig:
    prof = (rc.training_profile or "balanced").lower()
    if rc.vae_epochs <= 0 or rc.gan_epochs <= 0 or rc.batch_size <= 0:
        if prof == "fast":
            rc.vae_epochs = rc.vae_epochs or 3
            rc.gan_epochs = rc.gan_epochs or 5
            rc.batch_size = rc.batch_size or 128
        elif prof == "thorough":
            rc.vae_epochs = rc.vae_epochs or 30
            rc.gan_epochs = rc.gan_epochs or 100
            rc.batch_size = rc.batch_size or 512
        else:
            rc.vae_epochs = rc.vae_epochs or 10
            rc.gan_epochs = rc.gan_epochs or 30
            rc.batch_size = rc.batch_size or 256
    return rc

def load_config(path: str) -> RunConfig:
    import yaml
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    rc = RunConfig(**{k: v for k, v in y.items() if k in RunConfig.__dataclass_fields__})
    return _apply_training_profile(rc)
