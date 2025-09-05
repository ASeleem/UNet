"""UpSample block for U-Net architecture. Act as Decoder block."""
import torch
from torch import nn

from unet.components import DoubleConv

class UpSample(nn.Module):
    """UpSample block that applies ConvTranspose2d followed by DoubleConv.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass of the UpSample block.
        Args:
            x1 (torch.Tensor): Input tensor from the previous layer of shape (N, in_channels, H, W).
            x2 (torch.Tensor): Input tensor from the skip connection of shape (N, in_channels//2, H*2, W*2).
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H*2, W*2)."""
        x1 = self.up(x1)

        x = torch.cat([x1, x2], 1)

        return self.conv(x)
