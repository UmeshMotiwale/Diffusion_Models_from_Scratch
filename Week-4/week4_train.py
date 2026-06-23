import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from week4_model import UNet
from week4_schedular import NoiseScheduler
from week4_diffusion import p_losses, sample


SEED = 42
DATASET = "mnist"         
IMAGE_SHAPE = (1, 28, 28)
TIMESTEPS = 1000
SCHEDULE = "linear"        #linear/cosine
CHANNELS = (64, 128, 256, 512)
TIME_EMB_DIM = 256
BATCH_SIZE = 128
NUM_EPOCHS = 80
LR = 2e-4
SAMPLE_EVERY = 10
NUM_VIS_SAMPLES = 16
CKPT_PATH = "checkpoints/ddpm_mnist.pt"
RESULTS_DIR = "results"

torch.manual_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),])

DatasetClass = datasets.FashionMNIST if DATASET == "fashion_mnist" else datasets.MNIST
train_set = DatasetClass(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,)

print(f"Training on {DATASET}: {len(train_set)} images")

model = UNet(
    in_channels=IMAGE_SHAPE[0], out_channels=IMAGE_SHAPE[0],
    channels=CHANNELS, time_emb_dim=TIME_EMB_DIM,
).to(device)
scheduler = NoiseScheduler(timesteps=TIMESTEPS, schedule=SCHEDULE, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

def save_sample_grid(filename):
    model.eval()
    samples = sample(model, scheduler, IMAGE_SHAPE, NUM_VIS_SAMPLES, device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    grid = make_grid(samples, nrow=int(NUM_VIS_SAMPLES ** 0.5))
    save_image(grid, f"{RESULTS_DIR}/{filename}")
    model.train()
    print(f"  -> saved {RESULTS_DIR}/{filename}")

loss_history = []
best_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, running_batches = 0.0, 0

    for images, _labels in train_loader:
        images = images.to(device)
        t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=device).long()

        optimizer.zero_grad()
        loss = p_losses(model, scheduler, images, t)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_batches += 1

    epoch_loss = running_loss / running_batches
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | loss={epoch_loss:.5f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "loss": epoch_loss,
            "config": {
                "image_shape": IMAGE_SHAPE, "timesteps": TIMESTEPS, "schedule": SCHEDULE,
                "channels": CHANNELS, "time_emb_dim": TIME_EMB_DIM,
            },
        }, CKPT_PATH)
        print(f"  -> new best loss; checkpoint saved to {CKPT_PATH}")

    if epoch % SAMPLE_EVERY == 0 or epoch == NUM_EPOCHS:
        save_sample_grid(f"samples_epoch_{epoch}.png")

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.xlabel("epoch"); plt.ylabel("MSE noise-prediction loss"); plt.title("DDPM training loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=120)
print(f"Saved {RESULTS_DIR}/loss_curve.png")

model.eval()
final_samples = sample(model, scheduler, IMAGE_SHAPE, num_samples=64, device=device)
final_samples = (final_samples.clamp(-1, 1) + 1) / 2
grid = make_grid(final_samples, nrow=8)
save_image(grid, f"{RESULTS_DIR}/samples_final.png")
print(f"Saved {RESULTS_DIR}/samples_final.png")
print(f"\nDone. Best checkpoint: {CKPT_PATH} (loss={best_loss:.5f})")