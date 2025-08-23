import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.copula import GaussianCopula
from models.vae import TabularVAE
from models.gan import Generator, Critic, gradient_penalty
from models.autoregressive import CategoricalAR

class HybridGenerator:
    def __init__(self, cont_cols, cat_cols, device='cpu'):
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.device = device
        self.copula = GaussianCopula(cols=cont_cols)
        self.vae = None
        self.gan_G = None
        self.gan_D = None
        self.cat_ar = CategoricalAR(cat_cols=cat_cols, cont_cols=cont_cols)

    def fit(self, df: pd.DataFrame, epochs_vae=30, epochs_gan=200, batch=256):
        # 1) Fit copula on continuous
        self.copula.fit(df)
        # 2) Fit VAE on standardized continuous
        Xc = torch.tensor(df[self.cont_cols].to_numpy(), dtype=torch.float32)
        self.vae = TabularVAE(input_dim=Xc.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-5)
        loader = DataLoader(TensorDataset(Xc), batch_size=batch, shuffle=True)
        self.vae.train()
        for _ in range(epochs_vae):
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon, mu, logvar = self.vae(xb)
                loss, _ = TabularVAE.loss_fn(recon, xb, mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
        # 3) Train categorical AR on df (conditioned on cont)
        self.cat_ar.fit(df)
        # 4) Train WGAN-GP on residuals (VAE reconstruction residuals)
        with torch.no_grad():
            recon, _, _ = self.vae(Xc.to(self.device))
        resid = (Xc.to(self.device) - recon).cpu().numpy()
        R = torch.tensor(resid, dtype=torch.float32)
        z_dim = min(32, R.shape[1]*2)
        self.gan_G = Generator(z_dim=z_dim, x_dim=R.shape[1]).to(self.device)
        self.gan_D = Critic(x_dim=R.shape[1]).to(self.device)
        g_opt = torch.optim.Adam(self.gan_G.parameters(), lr=1e-4, betas=(0.5,0.9))
        d_opt = torch.optim.Adam(self.gan_D.parameters(), lr=1e-4, betas=(0.5,0.9))
        data_loader = DataLoader(TensorDataset(R), batch_size=128, shuffle=True)
        lambda_gp = 10.0
        for epoch in range(epochs_gan):
            for (rb,) in data_loader:
                rb = rb.to(self.device)
                # Train critic
                for _ in range(1):
                    z = torch.randn(rb.size(0), z_dim, device=self.device)
                    fake = self.gan_G(z)
                    d_real = self.gan_D(rb)
                    d_fake = self.gan_D(fake.detach())
                    gp = gradient_penalty(self.gan_D, rb, fake.detach())
                    d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp*gp
                    d_opt.zero_grad(); d_loss.backward(); d_opt.step()
                # Train generator
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                fake = self.gan_G(z)
                g_loss = -self.gan_D(fake).mean()
                g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        return self

    def sample(self, n: int, macro_row: dict = None) -> pd.DataFrame:
        # Sample continuous via copula → pass through VAE manifold → add GAN residual
        cont_df = self.copula.sample(n)
        Xc = torch.tensor(cont_df[self.cont_cols].to_numpy(), dtype=torch.float32).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            recon, _, _ = self.vae(Xc)
        # add synthetic residual from GAN
        z_dim = self.gan_G.net[0].in_features
        z = torch.randn(n, z_dim, device=self.device)
        r = self.gan_G(z)
        x_syn = (recon + r).cpu().numpy()
        cont_syn = pd.DataFrame(x_syn, columns=self.cont_cols)
        # Sample categoricals autoregressively conditioned on cont_syn
        cat_syn = self.cat_ar.sample(n, cont_syn)
        out = pd.concat([cont_syn, cat_syn], axis=1)
        return out