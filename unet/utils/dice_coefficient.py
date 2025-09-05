"""Utility functions for UNet model."""
import torch

def dice_coefficient(prediction, target, epsilon=1e-07):
    """Calculate the Dice Coefficient between prediction and target tensors.
    Args:
        prediction (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-07.
    Returns:
        float: Dice Coefficient score.
    """
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice
