import torch
from torch import nn as nn

class TabularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden: int = 128, beta=0.001, recon="mse", kl_warmup_steps=0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
        self.beta = beta
        self.recon = recon
        self.kl_warmup_steps = kl_warmup_steps
        self._step = 0

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        self._step += 1
        return recon, mu, logvar

    def loss_fn(self, recon, x, mu, logvar):
        if self.recon == "mae":
            recon_loss = nn.functional.l1_loss(recon, x, reduction='mean')
        else:
            recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.kl_warmup_steps > 0:
            w = min(1.0, self._step / float(self.kl_warmup_steps))
        else:
            w = 1.0
        return recon_loss + (self.beta * w) * kld, {'recon': recon_loss.item(), 'kld': kld.item(), 'kl_w': w}
