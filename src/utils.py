from typing import Tuple
import torch

if __name__ == '__main__':
    pass


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
