from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class ShockSpec:
    cont_mu_shift: dict = field(default_factory=dict)
    cont_scale:    dict = field(default_factory=dict)
    corr_shrink:   float = 0.0
    corr_pairs:    Dict[Tuple[str,str], float] = field(default_factory=dict)
    cat_logit_bias:dict = field(default_factory=dict)
    residual_scale: float = 1.0
    regime:        dict = field(default_factory=dict)

    def validate(self):
        self.corr_shrink = max(0.0, min(1.0, self.corr_shrink))
        return self
