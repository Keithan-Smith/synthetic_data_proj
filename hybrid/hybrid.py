import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from models.copula import GaussianCopula
from models.vae import TabularVAE
from models.gan import Generator, Critic, gradient_penalty
from models.autoregressive import CategoricalAR
from data.schemas import Schema, forward_transform, inverse_transform
from shocks.spec import ShockSpec
from privacy.dp import DPConfig, clip_and_noise_, rough_rdp_epsilon

class HybridGenerator:
    def __init__(self, cont_cols, cat_cols, schema: Schema = None, device='cuda'):
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.schema = schema
        self.device = device
        self.copula = GaussianCopula(cols=cont_cols)
        self.vae = None
        self.gan_G = None
        self.gan_D = None
        self.cat_ar = CategoricalAR(cat_cols=cat_cols, cont_cols=cont_cols)
        self.train_eps_log = {}

    def fit(self, df: pd.DataFrame, epochs_vae=20, epochs_gan=100, batch=256,
            privacy_cfg: dict = None, mine_cfg: dict = None):
        df_fit = df[self.cont_cols + self.cat_cols].dropna().reset_index(drop=True)

        if self.schema:
            df_fit_cont = forward_transform(df_fit[self.cont_cols], self.schema)
        else:
            df_fit_cont = df_fit[self.cont_cols].copy()

        # 1) Copula
        self.copula.fit(df_fit_cont)

        # 2) VAE (AMP + CUDA)
        Xc = torch.tensor(df_fit_cont.to_numpy(), dtype=torch.float32)
        self.vae = TabularVAE(input_dim=Xc.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        pin = ('cuda' in str(self.device))
        loader = DataLoader(TensorDataset(Xc), batch_size=batch, shuffle=True, pin_memory=pin, num_workers=2)
        self.vae.train()
        if pin: torch.backends.cudnn.benchmark = True
        scaler = GradScaler(enabled=pin)

        dp = None
        if privacy_cfg and privacy_cfg.get("enabled", False):
            dp = DPConfig(max_grad_norm=privacy_cfg.get("dp_max_grad_norm",1.0),
                          noise_multiplier=privacy_cfg.get("dp_noise_multiplier",0.0),
                          sample_rate=min(1.0, batch/max(1,Xc.shape[0])),
                          delta=privacy_cfg.get("dp_delta",1e-5))
        steps=0
        for _ in range(epochs_vae):
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                with autocast(enabled=('cuda' in str(self.device))):
                    recon, mu, logvar = self.vae(xb)
                    loss, _ = TabularVAE.loss_fn(recon, xb, mu, logvar)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if dp:
                    scaler.unscale_(opt); clip_and_noise_(self.vae, dp)
                scaler.step(opt); scaler.update(); steps+=1
        if dp: self.train_eps_log['vae_eps'] = rough_rdp_epsilon(dp.noise_multiplier, steps, dp.sample_rate, dp.delta)

        # 3) Categorical AR
        self.cat_ar.fit(df_fit)

        # 4) GAN on residuals (AMP + CUDA)
        with torch.no_grad():
            recon, _, _ = self.vae(Xc.to(self.device))
        resid = (Xc.to(self.device) - recon).cpu().numpy()
        R = torch.tensor(resid, dtype=torch.float32)
        z_dim = min(32, R.shape[1]*2)
        self.gan_G = Generator(z_dim=z_dim, x_dim=R.shape[1]).to(self.device)
        self.gan_D = Critic(x_dim=R.shape[1]).to(self.device)
        g_opt = torch.optim.Adam(self.gan_G.parameters(), lr=1e-4, betas=(0.5,0.9))
        d_opt = torch.optim.Adam(self.gan_D.parameters(), lr=1e-4, betas=(0.5,0.9))
        data_loader = DataLoader(TensorDataset(R), batch_size=128, shuffle=True, pin_memory=pin, num_workers=2)
        lambda_gp = 10.0
        steps=0
        g_scaler = GradScaler(enabled=pin); d_scaler = GradScaler(enabled=pin)

        for _ in range(epochs_gan):
            for (rb,) in data_loader:
                rb = rb.to(self.device, non_blocking=True)
                # Critic
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                with autocast(enabled=('cuda' in str(self.device))):
                    fake = self.gan_G(z)
                    d_real = self.gan_D(rb)
                    d_fake = self.gan_D(fake.detach())
                    gp = gradient_penalty(self.gan_D, rb, fake.detach())
                    d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp*gp
                d_opt.zero_grad(set_to_none=True)
                d_scaler.scale(d_loss).backward()
                if dp:
                    d_scaler.unscale_(d_opt); clip_and_noise_(self.gan_D, dp)
                d_scaler.step(d_opt); d_scaler.update()

                # Generator
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                with autocast(enabled=('cuda' in str(self.device))):
                    fake = self.gan_G(z)
                    g_loss = -self.gan_D(fake).mean()
                g_opt.zero_grad(set_to_none=True)
                g_scaler.scale(g_loss).backward()
                if dp:
                    g_scaler.unscale_(g_opt); clip_and_noise_(self.gan_G, dp)
                g_scaler.step(g_opt); g_scaler.update(); steps+=1
        if dp: self.train_eps_log['gan_eps'] = rough_rdp_epsilon(dp.noise_multiplier, steps, min(1.0, 128/max(1,R.shape[0])), dp.delta)
        return self

    def _apply_shocks_cont(self, x_df: pd.DataFrame, shock: ShockSpec):
        out = x_df.copy()
        for k, v in (shock.cont_mu_shift or {}).items():
            if k in out.columns: out[k] = out[k] + v
        for k, v in (shock.cont_scale or {}).items():
            if k in out.columns: out[k] = out[k] * v
        return out

    def _corr_with_shrink(self, R: np.ndarray, lam: float):
        k = R.shape[0]; I = np.eye(k); return (1-lam)*R + lam*I

    def sample(self, n: int, shock: ShockSpec = None) -> pd.DataFrame:
        shock = (shock or ShockSpec()).validate()
        corr = self.copula.corr.copy()
        if shock.corr_shrink > 0:
            corr = self._corr_with_shrink(corr, shock.corr_shrink)

        if self.schema:
            base = self.copula.sample(n, corr_override=corr)[self.cont_cols]
            base = forward_transform(base, self.schema)
        else:
            base = self.copula.sample(n, corr_override=corr)[self.cont_cols]

        base = self._apply_shocks_cont(base, shock)

        with torch.no_grad():
            Xc = torch.tensor(base.to_numpy(), dtype=torch.float32).to(self.device)
            recon, _, _ = self.vae(Xc)
            z_dim = self.gan_G.net[0].in_features
            z = torch.randn(n, z_dim, device=self.device)
            resid = self.gan_G(z) * float(shock.residual_scale)
            X = (recon + resid).cpu().numpy()
        cont_syn = pd.DataFrame(X, columns=self.cont_cols)

        cat_syn = self.cat_ar.sample(n, cont_syn, logit_bias=shock.cat_logit_bias)
        out = pd.concat([cont_syn, cat_syn], axis=1)

        if self.schema:
            out[self.cont_cols] = inverse_transform(out[self.cont_cols], self.schema)
        return out
