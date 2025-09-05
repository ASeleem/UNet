"""U-Net architecture for image segmentation."""
from torch import nn

from unet.components import DownSample, UpSample, DoubleConv

class UNet(nn.Module):
    """U-Net architecture.
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes."""
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_conv1 = DownSample(in_channels, 64)
        self.down_conv2 = DownSample(64, 128)
        self.down_conv3 = DownSample(128, 256)
        self.down_conv4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv1 = UpSample(1024, 512)
        self.up_conv2 = UpSample(512, 256)
        self.up_conv3 = UpSample(256, 128)
        self.up_conv4 = UpSample(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of the U-Net model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes, H, W)."""
        down_1, pooled_1 = self.down_conv1(x)
        down_2, pooled_2 = self.down_conv2(pooled_1)
        down_3, pooled_3 = self.down_conv3(pooled_2)
        down_4, pooled_4 = self.down_conv4(pooled_3)

        bottleneck = self.bottle_neck(pooled_4)

        up_1 = self.up_conv1(bottleneck, down_4)
        up_2 = self.up_conv2(up_1, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        return self.final_conv(up_4)
