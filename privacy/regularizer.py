# privacy/regularizer.py
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class PrivacyRegConfig:
    enabled: bool = True
    weight: float = 1.0         # λ_privacy in total loss
    alpha_vae: float = 1.0      # weight on VAE reconstruction-gap
    alpha_gan: float = 1.0      # weight on D confidence-gap
    holdout_frac: float = 0.1   # split real data into train/holdout for the gap

class PrivacyUtilityRegularizer(nn.Module):
    """
    Minimizes generalization gaps that drive MIA:
      - VAE gap: E[ReconLoss(train)] - E[ReconLoss(holdout)]
      - GAN gap: E[sigmoid(D(real_train))] - E[sigmoid(D(real_holdout))]
    Uses ReLU on gaps so only overfitting is penalized. Add scaled term to gen/enc/dec losses.
    """
    def __init__(self, cfg: PrivacyRegConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _mean_or_zero(t):
        return t.mean() if t is not None and t.numel() > 0 else torch.tensor(0.0, device=t.device if t is not None else "cpu")

    def forward(
        self,
        x_train_real,            # real batch from training split
        x_holdout_real,          # real batch from holdout split
        vae=None,                # object with methods encode/decode or forward returning recon + loss
        disc=None,               # discriminator returning logits for "real"
        recon_loss_fn=None       # function(x, vae)-> scalar per-sample recon loss (no reduction)
    ):
        if not self.cfg.enabled:
            return torch.tensor(0.0, device=x_train_real.device)

        loss_priv = 0.0

        # --- VAE gap ---
        if vae is not None and recon_loss_fn is not None:
            rec_tr = recon_loss_fn(x_train_real, vae)  # (B,)
            rec_ho = recon_loss_fn(x_holdout_real, vae)
            gap_rec = torch.relu(self._mean_or_zero(rec_tr) - self._mean_or_zero(rec_ho))
            loss_priv = loss_priv + self.cfg.alpha_vae * gap_rec

        # --- GAN gap via D confidence on real ---
        if disc is not None:
            logit_tr = disc(x_train_real)             # (B,) or (B,1) logits
            logit_ho = disc(x_holdout_real)
            p_tr = torch.sigmoid(logit_tr.squeeze(-1))
            p_ho = torch.sigmoid(logit_ho.squeeze(-1))
            gap_d = torch.relu(self._mean_or_zero(p_tr) - self._mean_or_zero(p_ho))
            loss_priv = loss_priv + self.cfg.alpha_gan * gap_d

        return self.cfg.weight * loss_priv
