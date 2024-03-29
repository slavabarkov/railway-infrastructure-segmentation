from typing import List
import torch
import numpy as np
import cv2

if __name__ == '__main__':
    pass


def get_batch_predictions(models: List[torch.nn.Module],
                          test_batch: torch.Tensor,
                          device: torch.device) -> torch.Tensor:
    """
    Returns the average predicted logits mask for each model in provided models list

    Parameters
    ----------
    models (List[torch.nn.Module]): list of loaded pytorch models
    test_batch (torch.Tensor): images batch to get predictions for
    device (str): device for pytorch tensors

    Returns
    -------
    test_output_batch (torch.Tensor): averaged predicted logits mask
    """
    for model in models:
        model.eval()
        model.to(device)

    test_batch = test_batch.to(device)
    test_output_batch = None

    with torch.inference_mode():
        for model in models:
            if test_output_batch is None:
                test_output_batch = model(test_batch)
            else:
                test_output_batch += model(test_batch)

        test_output_batch /= len(models)
        test_output_batch = torch.nn.Sigmoid()(test_output_batch)

    return test_output_batch


def upscale_mask(mask: np.ndarray, desired_height: int, desired_width: int) -> np.ndarray:
    """
    Reverses image size augmentations and returns upscaled mask with desired resolution

    Parameters
    ----------
    mask (np.ndarray): boolean segmentation mask
    desired_height (int): desired image height
    desired_width (int): desired image width

    Returns
    -------
    running_avg (float): Current running average
    """
    image_height, image_width = mask.shape[:2]

    ratio = desired_width / desired_height
    image_width_after_maxsize = image_width
    image_height_after_maxsize = round(image_width_after_maxsize / ratio)

    pad_top = int((image_height - image_height_after_maxsize) / 2.0)
    pad_bottom = image_height - image_height_after_maxsize - pad_top

    mask = mask[pad_top:-pad_bottom, :, :]
    mask = cv2.resize(mask, (desired_width, desired_height), cv2.INTER_NEAREST_EXACT)

    return mask


def convert_bool_mask_to_submission(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Converts the boolean mask to format required for submission

    Parameters
    ----------
    mask (np.ndarray): boolean segmentation mask
    threshold (float): threshold to convert logits to binary

    Returns
    -------
    output_mask (np.ndarray): segmentation mask in format required for submission
    """
    class_6_bool_mask = mask[:, :, 0] > threshold
    class_7_bool_mask = mask[:, :, 1] > threshold
    class_10_bool_mask = mask[:, :, 2] > threshold

    output_mask = np.zeros(mask.shape)
    output_mask[class_6_bool_mask] = 6
    output_mask[class_7_bool_mask] = 7
    output_mask[class_10_bool_mask] = 10

    return output_mask
