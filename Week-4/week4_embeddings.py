import math
import torch
import torch.nn as nn

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=device).float() / half_dim
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(emb)