"""
Wraps a clean image dataset (MNIST or CIFAR-10) and adds Gaussian noise
on-the-fly, returning (noisy_image, clean_image) pairs for the UNet to learn
to denoise.
Noise magnitude varies per-sample (drawn from a range) so the model learns to
handle a range of corruption levels, not just one fixed noise strength.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class NoisyDataset(Dataset):
    def __init__(self, base_dataset, noise_std_range=(0.1, 0.5)):
        
        self.base = base_dataset
        self.min_std, self.max_std = noise_std_range

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        clean, _label = self.base[idx]  # discard the class label — not needed
        std = torch.empty(1).uniform_(self.min_std, self.max_std).item()
        noise = torch.randn_like(clean) * std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)  # keep pixel values valid
        return noisy, clean


def get_mnist_noisy_datasets(root="./data", noise_std_range=(0.1, 0.5)):
    """Convenience loader: MNIST wrapped with noise, ready for DataLoader."""
    transform = transforms.ToTensor()  # keep pixels in [0,1] — simpler for MSE
    train_clean = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_clean = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_noisy = NoisyDataset(train_clean, noise_std_range)
    test_noisy = NoisyDataset(test_clean, noise_std_range)
    return train_noisy, test_noisy


def get_cifar10_noisy_datasets(root="./data", noise_std_range=(0.1, 0.5)):
    """Convenience loader: CIFAR-10 wrapped with noise, ready for DataLoader."""
    transform = transforms.ToTensor()
    train_clean = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_clean = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    train_noisy = NoisyDataset(train_clean, noise_std_range)
    test_noisy = NoisyDataset(test_clean, noise_std_range)
    return train_noisy, test_noisy

# train_ds, _ = get_mnist_noisy_datasets()

# noisy, clean = train_ds[0]
# print(noisy.shape, clean.shape)

# import matplotlib.pyplot as plt

# plt.subplot(1,2,1)
# plt.title("Noisy")
# plt.imshow(noisy.squeeze(), cmap="gray")

# plt.subplot(1,2,2)
# plt.title("Clean")
# plt.imshow(clean.squeeze(), cmap="gray")

# plt.show()