import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden: int = 128, ma_rate: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.register_buffer('ma_et', torch.tensor(0.0))
        self.initialized = False
        self.ma_rate = ma_rate

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        joint = torch.cat([x, z], dim=1)
        perm = torch.randperm(z.size(0), device=z.device)
        marg = torch.cat([x, z[perm]], dim=1)
        T_joint = self.net(joint)
        T_marg = self.net(marg)
        et = torch.exp(T_marg).mean()
        if not self.initialized:
            self.ma_et = et.detach(); self.initialized = True
        else:
            self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * et.detach()
        mi_est = T_joint.mean() - torch.log(self.ma_et + 1e-8)
        loss = -mi_est
        return loss, {'mi_est': mi_est.item()}
 