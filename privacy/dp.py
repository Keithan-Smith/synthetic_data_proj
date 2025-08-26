import torch
from dataclasses import dataclass

@dataclass
class DPConfig:
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.0
    sample_rate: float = 0.01
    delta: float = 1e-5

def clip_and_noise_(model: torch.nn.Module, cfg: DPConfig) -> None:
    grads = [p.grad for p in model.parameters() if (p.requires_grad and p.grad is not None)]
    if not grads: return
    device = grads[0].device
    total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
    clip_coef = min(1.0, cfg.max_grad_norm / (total_norm + 1e-12))
    for g in grads:
        g.mul_(clip_coef)
        if cfg.noise_multiplier > 0:
            noise = torch.normal(mean=0.0, std=cfg.noise_multiplier * cfg.max_grad_norm, size=g.shape, device=device)
            g.add_(noise)

def rough_rdp_epsilon(noise_multiplier: float, steps: int, sample_rate: float, delta: float = 1e-5) -> float:
    import math
    if noise_multiplier <= 0: return float('inf')
    q = sample_rate; sigma2 = noise_multiplier ** 2
    eps = steps * (q ** 2) / max(sigma2, 1e-12)
    return eps + math.log(1.0 / max(delta, 1e-12))
