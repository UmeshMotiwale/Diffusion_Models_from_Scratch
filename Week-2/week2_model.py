
import torch
import torch.nn as nn

from week2_blocks import DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=(64, 128, 256, 512), bilinear=True):

        """
        in_channels:  1 for grayscale (MNIST), 3 for RGB (CIFAR-10)
        out_channels: same as in_channels for a denoising autoencoder
        channels:     channel count at each encoder depth, e.g. (64,128,256,512)
                      means: stem->64, down1->128, down2->256, down3->512(bottleneck)
        """
        super().__init__()
        self.depth = len(channels)

        #Encoder
        self.stem = DoubleConv(in_channels, channels[0])
        self.down_blocks = nn.ModuleList([
            Down(channels[i], channels[i + 1])
            for i in range(self.depth - 1)
        ])

        #Decoder 
        self.up_blocks = nn.ModuleList([
            Up(channels[i + 1], channels[i], bilinear=bilinear)
            for i in reversed(range(self.depth - 1))
        ])

        #Output head
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Encoder save every feature map for skip connections
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

       
        skips = skips[:-1]

        #Decoder upsample + concat skip
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)


if __name__ == "__main__":
    # Quick shape sanity check — run this file directly: `python model.py`
    model = UNet(in_channels=1, out_channels=1, channels=(64, 128, 256, 512))
    dummy = torch.randn(2, 1, 28, 28)  # MNIST-sized batch
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == dummy.shape, "UNet output shape must match input shape!"
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Shape check passed ✓")