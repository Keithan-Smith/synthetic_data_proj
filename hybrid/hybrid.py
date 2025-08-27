import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import amp
from contextlib import nullcontext
import inspect

from models.copula import GaussianCopula
from models.vae import TabularVAE
from models.gan import Generator, Critic, gradient_penalty
from models.autoregressive import CategoricalAR
from data.schemas import Schema, forward_transform, inverse_transform
from shocks.spec import ShockSpec
from shocks.apply import build_corr_override, apply_cont_marginal_shocks
from privacy.dp import DPConfig, clip_and_noise_, rough_rdp_epsilon

# ---------------- AMP compatibility helpers ----------------
def _make_scaler(cuda_enabled: bool):
    """
    Return a GradScaler that works across torch versions.
    Falls back to torch.cuda.amp (old), and to a no-op if AMP is unavailable.
    """
    try:
        # Most torch versions accept this form
        return amp.GradScaler(enabled=cuda_enabled)
    except TypeError:
        try:
            from torch.cuda.amp import GradScaler as CudaGradScaler
            return CudaGradScaler(enabled=cuda_enabled)
        except Exception:
            class _NoopScaler:
                def scale(self, x): return x
                def unscale_(self, *args, **kwargs): pass
                def step(self, opt): opt.step()
                def update(self): pass
            return _NoopScaler()

def _autocast_ctx(cuda_enabled: bool):
    """
    Return an autocast context manager compatible with both new and old APIs.
    """
    if cuda_enabled:
        # prefer new API if available
        try:
            sig = inspect.signature(amp.autocast)
            if len(sig.parameters) >= 1:
                # new-style requires device string
                return amp.autocast('cuda')
            # older torch.amp with enabled kw
            return amp.autocast(enabled=True)
        except Exception:
            try:
                from torch.cuda.amp import autocast as cuda_autocast
                return cuda_autocast(enabled=True)
            except Exception:
                return nullcontext()
    else:
        # disabled context
        try:
            sig = inspect.signature(amp.autocast)
            if "enabled" in sig.parameters:
                return amp.autocast(enabled=False)
        except Exception:
            try:
                from torch.cuda.amp import autocast as cuda_autocast
                return cuda_autocast(enabled=False)
            except Exception:
                return nullcontext()
        return nullcontext()

# ---------------- Hybrid model ----------------
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

        # Work in transformed continuous space if schema provided
        if self.schema:
            df_fit_cont = forward_transform(df_fit[self.cont_cols], self.schema)
        else:
            df_fit_cont = df_fit[self.cont_cols].copy()

        # 1) Copula
        self.copula.fit(df_fit_cont)

        # 2) VAE (AMP-compatible)
        Xc = torch.tensor(df_fit_cont.to_numpy(), dtype=torch.float32)
        self.vae = TabularVAE(input_dim=Xc.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        cuda_on = ('cuda' in str(self.device))
        loader = DataLoader(TensorDataset(Xc), batch_size=batch, shuffle=True,
                            pin_memory=cuda_on, num_workers=2)
        self.vae.train()
        if cuda_on:
            torch.backends.cudnn.benchmark = True
        scaler = _make_scaler(cuda_on)

        dp = None
        if privacy_cfg and privacy_cfg.get("enabled", False):
            dp = DPConfig(max_grad_norm=privacy_cfg.get("dp_max_grad_norm",1.0),
                          noise_multiplier=privacy_cfg.get("dp_noise_multiplier",0.0),
                          sample_rate=min(1.0, batch/max(1,Xc.shape[0])),
                          delta=privacy_cfg.get("dp_delta",1e-5))
        steps = 0
        for _ in range(epochs_vae):
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                with _autocast_ctx(cuda_on):
                    recon, mu, logvar = self.vae(xb)
                    loss, _ = TabularVAE.loss_fn(recon, xb, mu, logvar)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if dp:
                    scaler.unscale_(opt); clip_and_noise_(self.vae, dp)
                scaler.step(opt); scaler.update(); steps += 1
        if dp:
            self.train_eps_log['vae_eps'] = rough_rdp_epsilon(
                dp.noise_multiplier, steps, dp.sample_rate, dp.delta
            )

        # 3) Categorical AR
        self.cat_ar.fit(df_fit)

        # 4) GAN on residuals (AMP-compatible)
        with torch.no_grad():
            recon, _, _ = self.vae(Xc.to(self.device))
        resid = (Xc.to(self.device) - recon).cpu().numpy()
        R = torch.tensor(resid, dtype=torch.float32)
        z_dim = min(32, R.shape[1]*2)
        self.gan_G = Generator(z_dim=z_dim, x_dim=R.shape[1]).to(self.device)
        self.gan_D = Critic(x_dim=R.shape[1]).to(self.device)
        g_opt = torch.optim.Adam(self.gan_G.parameters(), lr=1e-4, betas=(0.5,0.9))
        d_opt = torch.optim.Adam(self.gan_D.parameters(), lr=1e-4, betas=(0.5,0.9))
        data_loader = DataLoader(TensorDataset(R), batch_size=128, shuffle=True,
                                 pin_memory=cuda_on, num_workers=2)
        lambda_gp = 10.0
        steps = 0
        g_scaler = _make_scaler(cuda_on)
        d_scaler = _make_scaler(cuda_on)

        for _ in range(epochs_gan):
            for (rb,) in data_loader:
                rb = rb.to(self.device, non_blocking=True)
                # Critic
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                with _autocast_ctx(cuda_on):
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
                with _autocast_ctx(cuda_on):
                    fake = self.gan_G(z)
                    g_loss = -self.gan_D(fake).mean()
                g_opt.zero_grad(set_to_none=True)
                g_scaler.scale(g_loss).backward()
                if dp:
                    g_scaler.unscale_(g_opt); clip_and_noise_(self.gan_G, dp)
                g_scaler.step(g_opt); g_scaler.update(); steps += 1
        if dp:
            self.train_eps_log['gan_eps'] = rough_rdp_epsilon(
                dp.noise_multiplier, steps, min(1.0, 128/max(1,R.shape[0])), dp.delta
            )
        return self

    # ---------- shocks-aware sampling ----------
    def sample(self, n: int, shock: ShockSpec = None) -> pd.DataFrame:
        spec = (shock or ShockSpec()).validate()

        # 1) correlation override (shrink + targeted pairs) on continuous block
        C_base = self.copula.corr_matrix()
        C = build_corr_override(C_base, self.cont_cols, spec)

        # 2) draw base continuous block in TRANSFORMED space
        cont = self.copula.sample(n, corr_override=C)[self.cont_cols]

        # 3) marginal mean/scale shocks (still in transformed space)
        cont = apply_cont_marginal_shocks(cont, spec)

        # 4) add VAE+GAN residuals (scaled)
        with torch.no_grad():
            Xc = torch.tensor(cont.to_numpy(), dtype=torch.float32).to(self.device)
            recon, _, _ = self.vae(Xc)
            z_dim = self.gan_G.net[0].in_features
            z = torch.randn(n, z_dim, device=self.device)
            resid = self.gan_G(z) * float(spec.residual_scale)
            X = (recon + resid).cpu().numpy()
        cont_syn = pd.DataFrame(X, columns=self.cont_cols)

        # 5) categoricals with optional logit bias
        # If your CategoricalAR.sample ignores logit_bias, it will just be a no-op.
        cat_syn = self.cat_ar.sample(n, cont_syn, logit_bias=spec.cat_logit_bias)

        out = pd.concat([cont_syn, cat_syn], axis=1)

        # 6) invert transforms back to original scale at the very end
        if self.schema:
            out[self.cont_cols] = inverse_transform(out[self.cont_cols], self.schema)
        return out
