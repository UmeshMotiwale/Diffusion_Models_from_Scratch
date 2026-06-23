
import torch
import torch.nn.functional as F


def p_losses(model, scheduler, x_0, t, noise=None):
    #DDPM Algorithm 1,noise the image, predict and compute loss by MSE
    x_t, noise = scheduler.add_noise(x_0, t, noise=noise)
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def p_sample_step(model, scheduler, x_t, t_index):
    #DDPM Algorithm 2, one reverse step: x_t -> x_{t-1}
    device = x_t.device
    batch_size = x_t.shape[0]
    t_batch = torch.full((batch_size,), t_index, device=device, dtype=torch.long)

    beta_t = scheduler.betas[t_index].to(device)
    alpha_t = scheduler.alphas[t_index].to(device)
    sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t_index].to(device)

    predicted_noise = model(x_t, t_batch)
    model_mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)

    if t_index == 0:
        return model_mean
    noise = torch.randn_like(x_t)
    sigma_t = torch.sqrt(beta_t)
    return model_mean + sigma_t * noise

@torch.no_grad()
def sample(model, scheduler, image_shape, num_samples, device):
    #Full DDPM Algorithm 2: pure noise -> generated images.
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)
    for t_index in reversed(range(scheduler.timesteps)):
        x = p_sample_step(model, scheduler, x, t_index)
    return x

@torch.no_grad()
def sample_with_intermediates(model, scheduler, image_shape, num_samples, device, snapshot_steps=None):
    #returns snapshots at requested t value
    if snapshot_steps is None:
        snapshot_steps = []
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)
    snapshots = {}
    for t_index in reversed(range(scheduler.timesteps)):
        x = p_sample_step(model, scheduler, x, t_index)
        if t_index in snapshot_steps:
            snapshots[t_index] = x.clone()
    snapshots[0] = x.clone()
    return x, snapshots