from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class ShockSpec:
    cont_mu_shift: dict = field(default_factory=dict)
    cont_scale:    dict = field(default_factory=dict)
    corr_shrink:   float = 0.0
    corr_pairs:    Dict[Tuple[str, str], float] = field(default_factory=dict)
    cat_logit_bias: dict = field(default_factory=dict)
    residual_scale: float = 1.0
    regime: dict = field(default_factory=dict)

    def validate(self):
        try:
            self.corr_shrink = max(0.0, min(1.0, float(self.corr_shrink)))
        except Exception:
            self.corr_shrink = 0.0
        try:
            self.residual_scale = float(self.residual_scale or 1.0)
        except Exception:
            self.residual_scale = 1.0
        return self
