from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os, yaml

@dataclass
class RunConfig:
    data_source: str = ""
    uci_id: Optional[int] = None
    data_path: str = ""

    mode: str = "learned"
    adapter: str = "universal"
    column_map: str = ""
    domain_pack: str = ""
    output_dir: str = "outputs/run"
    seed: int = 42
    device: str = "cuda"

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

    shock: Dict[str, Any] = field(default_factory=dict)    # <-- pass-through from YAML
    eval_enabled: bool = True
    read_csv_kwargs: Dict[str, Any] = field(default_factory=dict)
    config_dir: str = ""   # filled by loader

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
    candidates = [
        os.path.expanduser(os.path.expandvars(path)),
        os.path.abspath(path),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path)),
    ]
    p = next((c for c in candidates if os.path.exists(c)), None)
    if p is None:
        raise FileNotFoundError(f"Config not found. Tried: {candidates}")
    with open(p, "r") as f:
        y = yaml.safe_load(f) or {}
    allowed = RunConfig.__dataclass_fields__.keys()
    rc = RunConfig(**{k: v for k, v in y.items() if k in allowed})
    rc.config_dir = os.path.dirname(p)
    return _apply_training_profile(rc)
