from typing import Tuple
import torch


def iou_coef(y_true: torch.Tensor,
             y_pred: torch.Tensor,
             threshold: float = 0.5,
             dimensions: Tuple[int, int] = (2, 3),
             epsilon: float = 1e-6) -> torch.Tensor:
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > threshold).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dimensions)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dimensions)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


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

        Args:
            n : value to add to the running average
        """
        self.total += n
        self.count += 1

    def __call__(self) -> float:
        """
        Returns current running average

        Returns:
            (float): Running average
        """
        return self.total / (self.count + 1e-15)
