import torch
import torch.nn as nn

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

  def forward(self, x):
        return self.net(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                     stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)

class Up(nn.Module):
   def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.channel_adjust = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.channel_adjust = nn.Identity()

        conv_in_channels = in_channels // 2 + out_channels  # upsampled + skip
        self.conv = DoubleConv(conv_in_channels, out_channels)

   def forward(self, x, skip):
        x = self.up(x)
        x = self.channel_adjust(x)

        # Handle off-by-one size mismatches between upsampled x and skip
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2,
                 diff_h // 2, diff_h - diff_h // 2],
            )

        x = torch.cat([skip, x], dim=1)  # concatenate along channel dim
        return self.conv(x)

