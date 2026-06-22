import math
import torch


class NoiseScheduler:
    def __init__(self, timesteps=1000, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        
        """
        timesteps:T, total number of diffusion steps (DDPM paper uses 1000)
        schedule:"linear" or "cosine"
        beta_start/beta_end: only used by the linear schedule
        """

        self.timesteps = timesteps
        self.schedule = schedule
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown_schedule: {schedule!r}. Use 'linear' or 'cosine'.")

        #Precomputed buffers
        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.999)

    def add_noise(self, x_0, t, noise=None):
        """
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,
            epsilon ~ N(0, I)
        """
        if isinstance(t, int):
            t = torch.full((x_0.shape[0],), t, device=x_0.device, dtype=torch.long)
        t = t.to(x_0.device)

        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ac = self.sqrt_alphas_cumprod.to(x_0.device)[t]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[t]

        #(B,1,1,1) so it broadcasts against (B,C,H,W)
        while sqrt_ac.dim() < x_0.dim():
            sqrt_ac = sqrt_ac.unsqueeze(-1)
            sqrt_omac = sqrt_omac.unsqueeze(-1)

        x_t = sqrt_ac * x_0 + sqrt_omac * noise
        return x_t, noise

    def snr(self):
        
        return self.alphas_cumprod / (1.0 - self.alphas_cumprod)