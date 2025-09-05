"""DownSample block for U-Net architecture. Act as Encoder block."""
from torch import nn

from unet.components import DoubleConv

class DownSample(nn.Module):
    """DownSample block that applies DoubleConv followed by MaxPooling.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward pass of the DownSample block.
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
        Returns:
            tuple: A tuple containing:
                - down (torch.Tensor): Output tensor after DoubleConv of shape (N, out_channels, H, W).
                - pooled (torch.Tensor): Output tensor after MaxPooling of shape (N, out_channels, H/2, W/2)."""
        down = self.conv(x)
        pooled = self.pool(down)

        return down, pooled
