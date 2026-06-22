
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from week3_schedular import NoiseScheduler

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTEPS = 1000
SHOW_AT = [0, 100, 250, 500, 750, 999]


def get_one_image():
    """Grab a single MNIST image, normalized to [-1, 1] (see Part 1 §5 for why)."""
    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),        # -> [-1, 1]
    ])
    try:
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        img, _ = ds[0]
    except Exception as e:
        print(f"Could not download MNIST ({e}); using a synthetic test image instead.")
        x = torch.linspace(-1, 1, 28).unsqueeze(0).repeat(28, 1)
        y = torch.linspace(-1, 1, 28).unsqueeze(1).repeat(1, 28)
        img = torch.clamp(1 - (x ** 2 + y ** 2), -1, 1).unsqueeze(0)
    return img.unsqueeze(0)  # add batch dim -> (1, 1, 28, 28)


def plot_noising_trajectory(scheduler, x_0):
    fig, axes = plt.subplots(1, len(SHOW_AT), figsize=(3 * len(SHOW_AT), 3.2))
    torch.manual_seed(0)
    for ax, t in zip(axes, SHOW_AT):
        x_t, _ = scheduler.add_noise(x_0, t)
        img = x_t[0, 0].clamp(-1, 1)
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t = {t}")
        ax.axis("off")
    plt.suptitle(f"Forward diffusion trajectory ({scheduler.schedule} schedule)")
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/noising_trajectory.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_snr_comparison():
    linear = NoiseScheduler(timesteps=TIMESTEPS, schedule="linear")
    cosine = NoiseScheduler(timesteps=TIMESTEPS, schedule="cosine")

    plt.figure(figsize=(7, 5))
    plt.plot(linear.snr().numpy(), label="linear")
    plt.plot(cosine.snr().numpy(), label="cosine")
    plt.yscale("log")
    plt.xlabel("timestep t")
    plt.ylabel(r"SNR = $\bar\alpha_t / (1-\bar\alpha_t)$  (log scale)")
    plt.title("Signal-to-noise ratio: linear vs. cosine schedule")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/linear_vs_cosine.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def plot_final_noise_distribution(scheduler, x_0_batch):
    """Histogram of x_T values — should look like standard normal."""
    x_T, _ = scheduler.add_noise(x_0_batch, scheduler.timesteps - 1)
    values = x_T.flatten().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=80, density=True, alpha=0.7, label="x_T pixel values")
    xs = torch.linspace(-4, 4, 200)
    gaussian = torch.exp(-xs ** 2 / 2) / (2 * 3.14159265) ** 0.5
    plt.plot(xs.numpy(), gaussian.numpy(), "r--", label="N(0,1) reference")
    plt.title(f"x_T distribution ({scheduler.schedule})  |  mean={values.mean():.3f}, std={values.std():.3f}")
    plt.legend()
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/final_noise_distribution.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path} (mean={values.mean():.4f}, std={values.std():.4f}, should be ~0, ~1)")


if __name__ == "__main__":
    scheduler = NoiseScheduler(timesteps=TIMESTEPS, schedule="linear")

    x_0 = get_one_image()
    plot_noising_trajectory(scheduler, x_0)
    plot_snr_comparison()

    x_0_batch = x_0.repeat(256, 1, 1, 1)
    plot_final_noise_distribution(scheduler, x_0_batch)