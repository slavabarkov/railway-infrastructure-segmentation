from typing import List, Dict, Any
import torch
import numpy as np
import cv2


def get_batch_predictions(models: List[torch.nn.Module],
                          batch_data: Dict[str, Any],
                          device: str = 'cuda') -> torch.Tensor:
    """
    Returns the average predicted logits mask for each model in provided models list

    Parameters
    ----------
    models (List[torch.nn.Module]): list of loaded pytorch models
    batch_data (dict): current batch data, expects batches of size 1
    device (str): device for pytorch tensors

    Returns
    -------
    test_output_batch (torch.Tensor): averaged predicted logits mask
    """
    test_batch = batch_data['image']
    test_batch = test_batch.to(device)
    test_output_batch = None

    for model in models:
        model.eval()
        with torch.inference_mode():
            if test_output_batch is None:
                test_output_batch = model(test_batch)
            else:
                test_output_batch += model(test_batch)
                test_output_batch /= 2

        test_output_batch = torch.nn.Sigmoid()(test_output_batch)
    return test_output_batch


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
