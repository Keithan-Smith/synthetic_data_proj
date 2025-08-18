import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim)
        )
    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, device=real.device)
    inter = alpha*real + (1-alpha)*fake
    inter.requires_grad_(True)
    score = critic(inter)
    grads = torch.autograd.grad(outputs=score, inputs=inter,
                                grad_outputs=torch.ones_like(score),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp
