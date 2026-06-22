"""
Trains the UNet as a denoising autoencoder:
    noisy_image -> UNet -> predicted_clean_image
Loss: MSE between prediction and the true clean image.        
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from week2_model import UNet
from week2_datasheet import get_mnist_noisy_datasets

if __name__ == "__main__":
    SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LR = 1e-3
    VAL_FRACTION = 0.1
    NOISE_STD_RANGE = (0.1, 0.5)
    VIS_EVERY = 5          # save a sample grid every N epochs
    CKPT_PATH = "best_model.pt"
    RESULTS_DIR = "results"

    torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Data 
    full_train, test_set = get_mnist_noisy_datasets(noise_std_range=NOISE_STD_RANGE)

    val_size = int(len(full_train) * VAL_FRACTION)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


    vis_noisy, vis_clean = next(iter(val_loader))
    vis_noisy, vis_clean = vis_noisy[:8].to(device), vis_clean[:8].to(device)

    #Model
    model = UNet(in_channels=1, out_channels=1, channels=(64, 128, 256, 512)).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        total_loss, total_samples = 0.0, 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            total_loss += loss.item() * noisy.size(0)
            total_samples += noisy.size(0)
        return total_loss / total_samples


    @torch.no_grad()
    def save_visualization(model, epoch):
        """Save a (noisy | denoised | clean) triplet grid for a fixed val batch."""
        model.eval()
        pred = model(vis_noisy).clamp(0, 1)

        n = vis_noisy.size(0)
        fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
        for i in range(n):
            axes[0, i].imshow(vis_noisy[i, 0].cpu(), cmap="gray")
            axes[1, i].imshow(pred[i, 0].cpu(), cmap="gray")
            axes[2, i].imshow(vis_clean[i, 0].cpu(), cmap="gray")
            for row in range(3):
                axes[row, i].axis("off")
        axes[0, 0].set_ylabel("noisy", fontsize=10)
        axes[0, 0].axis("on"); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
        axes[1, 0].set_ylabel("denoised", fontsize=10)
        axes[1, 0].axis("on"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
        axes[2, 0].set_ylabel("ground truth", fontsize=10)
        axes[2, 0].axis("on"); axes[2, 0].set_xticks([]); axes[2, 0].set_yticks([])

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/samples_epoch_{epoch}.png", dpi=120)
        plt.close(fig)


    #Training loop 
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, running_samples = 0.0, 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            running_samples += noisy.size(0)

        train_loss = running_loss / running_samples
        val_loss = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, CKPT_PATH)
            print(f"  -> new best val_loss; checkpoint saved to {CKPT_PATH}")

        if epoch % VIS_EVERY == 0 or epoch == NUM_EPOCHS:
            save_visualization(model, epoch)
            print(f"  -> saved {RESULTS_DIR}/samples_epoch_{epoch}.png")

    #Loss curve
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Denoising loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=120)
    print(f"Saved {RESULTS_DIR}/loss_curve.png")

    #Final test evaluation
    ckpt = torch.load(CKPT_PATH, map_location=device)
    fresh_model = UNet(in_channels=1, out_channels=1, channels=(64, 128, 256, 512)).to(device)
    fresh_model.load_state_dict(ckpt["model_state"])
    test_loss = evaluate(fresh_model, test_loader)
    print(f"\nLoaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")
    print(f"FINAL TEST LOSS (MSE): {test_loss:.5f}")


