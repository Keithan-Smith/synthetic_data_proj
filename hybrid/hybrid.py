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
from privacy.mi import MINE  # <-- MINE

# ---------------- small type helpers (robust to YAML strings) ----------------
def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return False

def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

# ---------------- AMP compatibility helpers ----------------
def _make_scaler(cuda_enabled: bool):
    """
    Return a GradScaler that works across torch versions.
    Falls back to torch.cuda.amp (old), and to a no-op if AMP is unavailable.
    """
    try:
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
        try:
            sig = inspect.signature(amp.autocast)
            if len(sig.parameters) >= 1:
                return amp.autocast('cuda')
            return amp.autocast(enabled=True)
        except Exception:
            try:
                from torch.cuda.amp import autocast as cuda_autocast
                return cuda_autocast(enabled=True)
            except Exception:
                return nullcontext()
    else:
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

# ---------------- simple RBF-MMD regularizer ----------------
def _rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Unbiased squared MMD with RBF kernel between batches x and y.
    x, y: [B, D] (grad flows through x; y can be detached)
    """
    x = x.float()
    y = y.float()
    bx = x.size(0)
    by = y.size(0)
    if bx < 2 or by < 2:
        return x.new_tensor(0.0)

    def _rbf(a, b):
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True)
        ab = a @ b.t()
        dist2 = aa - 2 * ab + bb.t()
        return torch.exp(-dist2 / (2.0 * (sigma ** 2)))

    Kxx = _rbf(x, x)
    Kyy = _rbf(y, y)
    Kxy = _rbf(x, y)

    Kxx = (Kxx.sum() - Kxx.diag().sum()) / (bx * (bx - 1))
    Kyy = (Kyy.sum() - Kyy.diag().sum()) / (by * (by - 1))
    Kxy = Kxy.mean()
    return Kxx + Kyy - 2.0 * Kxy

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
        self.train_logs = {}            # scalar logs (e.g., mean MMDs, MI estimates)
        self.priv_cfg = None
        self.priv_reg = None
        self.holdout_idx = None
        self._cont_stats = None

    def fit(self, df: pd.DataFrame, epochs_vae=20, epochs_gan=100, batch=256,
            privacy_cfg: dict = None, mine_cfg: dict = None, privacy_reg_cfg: dict | None = None):
        df_fit = df[self.cont_cols + self.cat_cols].dropna().reset_index(drop=True)

        # Work in transformed continuous space if schema provided
        if self.schema:
            df_fit_cont = forward_transform(df_fit[self.cont_cols], self.schema)
        else:
            df_fit_cont = df_fit[self.cont_cols].copy()

        mu = df_fit_cont.mean(axis=0)
        sd = df_fit_cont.std(axis=0).replace(0.0, 1.0)
        self._cont_stats = {c: (float(mu[c]), float(sd[c])) for c in self.cont_cols}

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

        # --- DP config (optional, robust to YAML strings) ---
        dp = None
        priv_cfg = privacy_cfg or {}
        if _as_bool(priv_cfg.get("enabled", False)):
            dp = DPConfig(
                max_grad_norm=_as_float(priv_cfg.get("dp_max_grad_norm", 1.0), 1.0),
                noise_multiplier=_as_float(priv_cfg.get("dp_noise_multiplier", 0.0), 0.0),
                sample_rate=float(min(1.0, batch / max(1, Xc.shape[0]))),
                delta=_as_float(priv_cfg.get("dp_delta", 1e-5), 1e-5),
            )

        # --- Privacy–utility regularizer config (optional) ---
        reg = privacy_reg_cfg or {}
        reg_enabled = _as_bool(reg.get("enabled", False))
        reg_mode = str(reg.get("mode", "attractive")).lower()          # 'attractive' or 'repulsive'
        reg_sigma = _as_float(reg.get("mmd_bandwidth", 1.0), 1.0)
        reg_lam_g = _as_float(reg.get("lambda_mmd_gan", reg.get("lambda_mmd", 0.0)), 0.0)
        reg_lam_v = _as_float(reg.get("lambda_mmd_vae", max(0.0, 0.25 * reg_lam_g)), 0.0)

        # --- MINE config (optional) ---
        mine_cfg = mine_cfg or {}
        # accept either a dedicated mine_cfg OR keys in priv_cfg
        mine_enabled = _as_bool(mine_cfg.get("enabled", priv_cfg.get("mine_enabled", False)))
        mine_lambda = _as_float(mine_cfg.get("lambda", priv_cfg.get("mine_lambda", 0.0)), 0.0)

        self.priv_cfg = privacy_cfg
        self.priv_reg = reg

        # --- VAE training ---
        steps = 0
        mmd_vae_sum, mmd_vae_cnt = 0.0, 0

        # MINE for VAE (x,z)
        if mine_enabled and mine_lambda > 0.0:
            latent_dim = self.vae.mu.out_features
            mine_vae = MINE(x_dim=Xc.shape[1], z_dim=latent_dim, hidden=128, ma_rate=0.01).to(self.device)
            mine_vae_opt = torch.optim.Adam(mine_vae.parameters(), lr=1e-4)
        else:
            mine_vae, mine_vae_opt = None, None

        for _ in range(epochs_vae):
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)

                # ---- (1) MINE critic step on DETACHED tensors (no VAE graph) ----
                if mine_vae is not None:
                    with torch.no_grad():
                        mu_d, logvar_d = self.vae.encode(xb)              # no graph
                        z_d = self.vae.reparam(mu_d, logvar_d)             # no graph
                    mine_vae_opt.zero_grad(set_to_none=True)
                    with _autocast_ctx(cuda_on):
                        mine_loss_train, _ = mine_vae(x=xb.detach(), z=z_d.detach())  # loss=-MI
                    mine_loss_train.backward()
                    mine_vae_opt.step()

                # ---- (2) VAE step: single backward; MINE is FROZEN here ----
                with _autocast_ctx(cuda_on):
                    recon, mu_t, logvar_t = self.vae(xb)
                    vae_loss, _ = TabularVAE.loss_fn(recon, xb, mu_t, logvar_t)
                    total_loss = vae_loss

                    # MMD (optional)
                    if reg_enabled and reg_lam_v > 0.0:
                        mmd_v = _rbf_mmd2(recon, xb.detach(), sigma=reg_sigma)
                        total_loss = total_loss + reg_lam_v * mmd_v
                    else:
                        mmd_v = None

                    # MINE penalty (reduce MI): add λ * (-loss_eval)
                    if mine_vae is not None and mine_lambda > 0.0:
                        for p in mine_vae.parameters():
                            p.requires_grad_(False)
                        z_eval = self.vae.reparam(mu_t, logvar_t)   # grad flows to encoder
                        mine_loss_eval, _ = mine_vae(x=xb.detach(), z=z_eval)
                        total_loss = total_loss + mine_lambda * (-mine_loss_eval)
                    else:
                        mine_loss_eval = None

                opt.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()  # single backward through VAE
                if dp:
                    scaler.unscale_(opt); clip_and_noise_(self.vae, dp)
                scaler.step(opt); scaler.update(); steps += 1

                if mine_vae is not None:
                    for p in mine_vae.parameters():
                        p.requires_grad_(True)

                if mmd_v is not None and torch.isfinite(mmd_v):
                    mmd_vae_sum += float(mmd_v.detach().cpu())
                    mmd_vae_cnt += 1

                # logging MI estimate (optional)
                if mine_loss_eval is not None:
                    mi_val = float((-mine_loss_eval).detach().cpu())
                    self.train_logs['mi_est_vae_sum'] = self.train_logs.get('mi_est_vae_sum', 0.0) + mi_val
                    self.train_logs['mi_est_vae_cnt'] = self.train_logs.get('mi_est_vae_cnt', 0) + 1

        if dp:
            if dp.noise_multiplier > 0:
                self.train_eps_log['vae_eps'] = rough_rdp_epsilon(
                    dp.noise_multiplier, steps, dp.sample_rate, dp.delta
                )
            else:
                self.train_eps_log['vae_eps'] = float("inf")
        if mmd_vae_cnt > 0:
            self.train_logs['mmd_vae_mean'] = mmd_vae_sum / mmd_vae_cnt
        if self.train_logs.get('mi_est_vae_cnt', 0) > 0:
            self.train_logs['mi_est_vae_mean'] = self.train_logs['mi_est_vae_sum'] / self.train_logs['mi_est_vae_cnt']

        # 3) Categorical AR
        self.cat_ar.fit(df_fit)

        # 4) GAN on residuals (AMP-compatible)
        with torch.no_grad():
            recon_all, _, _ = self.vae(Xc.to(self.device))
        resid = (Xc.to(self.device) - recon_all).cpu().numpy()
        R = torch.tensor(resid, dtype=torch.float32)
        z_dim = min(32, R.shape[1] * 2)
        self.gan_G = Generator(z_dim=z_dim, x_dim=R.shape[1]).to(self.device)
        self.gan_D = Critic(x_dim=R.shape[1]).to(self.device)
        g_opt = torch.optim.Adam(self.gan_G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.gan_D.parameters(), lr=1e-4, betas=(0.5, 0.9))
        data_loader = DataLoader(TensorDataset(R), batch_size=128, shuffle=True,
                                 pin_memory=cuda_on, num_workers=2)
        lambda_gp = 10.0
        steps = 0
        g_scaler = _make_scaler(cuda_on)
        d_scaler = _make_scaler(cuda_on)

        # MINE for GAN (real_resid, fake_resid)
        if mine_enabled and mine_lambda > 0.0:
            mine_gan = MINE(x_dim=R.shape[1], z_dim=R.shape[1], hidden=128, ma_rate=0.01).to(self.device)
            mine_gan_opt = torch.optim.Adam(mine_gan.parameters(), lr=1e-4)
        else:
            mine_gan, mine_gan_opt = None, None

        mmd_g_sum, mmd_g_cnt = 0.0, 0
        for _ in range(epochs_gan):
            for (rb,) in data_loader:
                rb = rb.to(self.device, non_blocking=True)

                # --- Critic ---
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                with _autocast_ctx(cuda_on):
                    fake = self.gan_G(z)
                    d_real = self.gan_D(rb)
                    d_fake = self.gan_D(fake.detach())
                    gp = gradient_penalty(self.gan_D, rb, fake.detach())
                    d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
                d_opt.zero_grad(set_to_none=True)
                d_scaler.scale(d_loss).backward()
                if dp:
                    d_scaler.unscale_(d_opt); clip_and_noise_(self.gan_D, dp)
                d_scaler.step(d_opt); d_scaler.update()

                # --- MINE critic update on detached tensors ---
                if mine_gan is not None:
                    mine_gan_opt.zero_grad(set_to_none=True)
                    with _autocast_ctx(cuda_on):
                        mine_loss_g_train, _ = mine_gan(x=rb.detach(), z=fake.detach())  # loss=-MI
                    mine_loss_g_train.backward()
                    mine_gan_opt.step()

                # --- Generator ---
                z = torch.randn(rb.size(0), z_dim, device=self.device)
                with _autocast_ctx(cuda_on):
                    fake = self.gan_G(z)
                    g_loss = -self.gan_D(fake).mean()

                    # MMD (optional)
                    if reg_enabled and reg_lam_g > 0.0:
                        mmd_g = _rbf_mmd2(fake, rb.detach(), sigma=reg_sigma)
                        if reg_mode == "repulsive":
                            g_loss = g_loss - reg_lam_g * mmd_g
                        else:
                            g_loss = g_loss + reg_lam_g * mmd_g
                    else:
                        mmd_g = None

                    # MINE penalty (reduce MI(real,fake)): add λ * (-loss_eval) with MINE frozen
                    if mine_gan is not None and mine_lambda > 0.0:
                        for p in mine_gan.parameters():
                            p.requires_grad_(False)
                        mine_loss_g_eval, _ = mine_gan(x=rb.detach(), z=fake)  # grad flows to G via fake
                        g_loss = g_loss + mine_lambda * (-mine_loss_g_eval)
                    else:
                        mine_loss_g_eval = None

                g_opt.zero_grad(set_to_none=True)
                g_scaler.scale(g_loss).backward()
                if dp:
                    g_scaler.unscale_(g_opt); clip_and_noise_(self.gan_G, dp)
                g_scaler.step(g_opt); g_scaler.update(); steps += 1

                if mine_gan is not None:
                    for p in mine_gan.parameters():
                        p.requires_grad_(True)

                if mmd_g is not None and torch.isfinite(mmd_g):
                    mmd_g_sum += float(mmd_g.detach().cpu())
                    mmd_g_cnt += 1

                if mine_loss_g_eval is not None:
                    mi_val_g = float((-mine_loss_g_eval).detach().cpu())
                    self.train_logs['mi_est_gan_sum'] = self.train_logs.get('mi_est_gan_sum', 0.0) + mi_val_g
                    self.train_logs['mi_est_gan_cnt'] = self.train_logs.get('mi_est_gan_cnt', 0) + 1

        if dp:
            if dp.noise_multiplier > 0:
                self.train_eps_log['gan_eps'] = rough_rdp_epsilon(
                    dp.noise_multiplier, steps, min(1.0, 128 / max(1, R.shape[0])), dp.delta
                )
            else:
                self.train_eps_log['gan_eps'] = float("inf")
        if mmd_g_cnt > 0:
            self.train_logs['mmd_gan_mean'] = mmd_g_sum / mmd_g_cnt
        if self.train_logs.get('mi_est_gan_cnt', 0) > 0:
            self.train_logs['mi_est_gan_mean'] = self.train_logs['mi_est_gan_sum'] / self.train_logs['mi_est_gan_cnt']

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
        cat_syn = self.cat_ar.sample(n, cont_syn, logit_bias=spec.cat_logit_bias)

        out = pd.concat([cont_syn, cat_syn], axis=1)

        # 6) invert transforms back to original scale at the very end
        if self._cont_stats is not None:
            k = 8.0
            for c in self.cont_cols:
                m, s = self._cont_stats[c]
                cont[c] = cont[c].clip(m - k*s, m + k*s)
        if self.schema:
            out[self.cont_cols] = inverse_transform(out[self.cont_cols], self.schema)
        return out
