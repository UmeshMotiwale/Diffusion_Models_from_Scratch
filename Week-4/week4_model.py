#Time-conditioned UNet for noise prediction: eps_theta(x_t, t) -> predicted noise.
import torch
import torch.nn as nn
import torch.nn.functional as F

from week4_embeddings import TimeEmbedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.act(self.bn1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.residual_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.res = ResBlock(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.downsample(x)
        return self.res(x, t_emb)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode= "bilinear", align_corners=True)
            self.channel_adjust = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.channel_adjust = nn.Identity()
        self.res = ResBlock(in_channels // 2 + out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = self.channel_adjust(x)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]

        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)

        return self.res(x, t_emb)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=(64, 128, 256, 512),
                 time_emb_dim=256, bilinear=True):
        super().__init__()
        self.depth = len(channels)
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.stem = ResBlock(in_channels, channels[0], time_emb_dim)
        self.down_blocks = nn.ModuleList([
            Down(channels[i], channels[i + 1], time_emb_dim)

            for i in range(self.depth - 1)])
        self.up_blocks = nn.ModuleList([
            Up(channels[i + 1], channels[i], time_emb_dim, bilinear=bilinear)

            for i in reversed(range(self.depth - 1))])
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        skips = []
        x = self.stem(x, t_emb)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x, t_emb)
            skips.append(x)
        skips = skips[:-1]

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, t_emb)

        return self.out_conv(x)