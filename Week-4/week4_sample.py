import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from week4_model import UNet
from week4_schedular import NoiseScheduler
from week4_diffusion import sample, sample_with_intermediates

CKPT_PATH = "checkpoints/ddpm_mnist.pt"
RESULTS_DIR = "results"
NUM_SAMPLES = 64

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt = torch.load(CKPT_PATH, map_location=device)
cfg = ckpt["config"]
print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.5f})")

model = UNet(
    in_channels=cfg["image_shape"][0], out_channels=cfg["image_shape"][0],
    channels=cfg["channels"], time_emb_dim=cfg["time_emb_dim"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

scheduler = NoiseScheduler(timesteps=cfg["timesteps"], schedule=cfg["schedule"], device=device)

samples = sample(model, scheduler, cfg["image_shape"], NUM_SAMPLES, device)
samples = (samples.clamp(-1, 1) + 1) / 2
grid = make_grid(samples, nrow=8)
save_image(grid, f"{RESULTS_DIR}/samples_final.png")
print(f"Saved {RESULTS_DIR}/samples_final.png")

snapshot_steps = [999, 750, 500, 250, 100, 0]
_, snapshots = sample_with_intermediates(
    model, scheduler, cfg["image_shape"], num_samples=1, device=device,
    snapshot_steps=snapshot_steps,
)

fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(3 * len(snapshot_steps), 3.2))
for ax, t_idx in zip(axes, snapshot_steps):
    img = ((snapshots[t_idx][0, 0].clamp(-1, 1) + 1) / 2).cpu()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t ≈ {t_idx}")
    ax.axis("off")
plt.suptitle("Reverse diffusion: pure noise -> generated image")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/denoising_trajectory.png", dpi=120)
print(f"Saved {RESULTS_DIR}/denoising_trajectory.png")