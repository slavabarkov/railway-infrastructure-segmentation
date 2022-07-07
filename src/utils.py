from typing import Tuple
import torch
import numpy as np
import cv2


class RunningAverage():
    """
    Class to save and update the running average
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, n: float) -> None:
        """
        Updates running average with new value

        Parameters
        ----------
        n (float): value to add to the running average
        """
        self.total += n
        self.count += 1

    def __call__(self) -> float:
        """
        Returns current running average

        Returns
        -------
        running_avg (float): Current running average
        """
        running_avg = self.total / (self.count + 1e-15)
        return running_avg


def iou_coef(y_true: torch.Tensor,
             y_pred: torch.Tensor,
             threshold: float = 0.5,
             dimensions: Tuple[int, int] = (2, 3),
             epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculate the IoU coefficient

    Parameters
    ----------
    y_true (torch.Tensor): actual segmentation masks
    y_pred (torch.Tensor): model predictions
    threshold (float): threshold to convert logits to binary
    dimensions (tuple): dimensions used to calculate the metric
    epsilon (float): epsilon value to prevent division by zero

    Returns
    -------
    iou (float): IoU coefficient
    """
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > threshold).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dimensions)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dimensions)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def upscale_mask(mask: np.array, desired_height: int, desired_width: int) -> np.array:
    """
    Reverses image size augmentations and returns upscaled mask with desired resolution

    Parameters
    ----------
    mask (np.array): boolean segmentation mask
    desired_height (int): desired image height
    desired_width (int): desired image width

    Returns
    -------
    running_avg (float): Current running average
    """
    image_height, image_width = mask.shape[:2]

    ratio = desired_width / desired_height
    image_width_after_maxsize = image_width
    image_height_after_maxsize = round(image_width_after_maxsize * ratio)

    pad_top = int((image_height - image_height_after_maxsize) / 2.0)
    pad_bottom = image_height - image_height_after_maxsize - pad_top

    mask = mask[pad_top:-pad_bottom, :, :]
    mask = cv2.resize(mask, (desired_width, desired_height), cv2.INTER_NEAREST_EXACT)

    return mask


def convert_bool_mask_to_submission(mask: np.array, threshold: float = 0.5) -> np.array:
    """
    Converts the boolean mask to format required for submission

    Parameters
    ----------
    mask (np.array): boolean segmentation mask
    threshold (float): threshold to convert logits to binary

    Returns
    -------
    output_mask (np.array): segmentation mask in format required for submission
    """
    class_6_bool_mask = mask[:, :, 0] > threshold
    class_7_bool_mask = mask[:, :, 1] > threshold
    class_10_bool_mask = mask[:, :, 2] > threshold

    output_mask = np.zeros(mask.shape)
    output_mask[class_6_bool_mask] = 6
    output_mask[class_7_bool_mask] = 7
    output_mask[class_10_bool_mask] = 10

    return output_mask
