"""A module consisting of two consecutive convolutional layers each followed by a ReLU activation."""
from torch import nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_op = nn.Sequential(
            # First convolution block
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Second convolution block
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass of the DoubleConv block.
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)."""
        return self.conv_op(x)
