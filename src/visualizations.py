import albumentations as A
import cv2
import numpy as np

if __name__ == '__main__':
    pass


def visualize_augmentations(image_path: str,
                            mask_path: str,
                            transforms: A.core.composition.Compose) -> np.ndarray:
    """
    Applies the specified augmentation to images and masks and returns the visualization

    Parameters
    ----------
    image_path (str): Image path
    mask_path (str): Mask path
    transforms (albumentations.Compose): Albumentations transforms

    Returns
    -------
    blend (np.ndarray): Image with mask and applied augmentations
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32')
    image /= 255.

    mask = cv2.imread(mask_path)
    mask[:, :, 0] = mask[:, :, 0] == 6
    mask[:, :, 1] = mask[:, :, 1] == 7
    mask[:, :, 2] = mask[:, :, 2] == 10
    mask = mask.astype('float32')

    result = {'image': image, 'mask': mask}
    result = transforms(**result)

    blend = np.clip(result['image'] + result['mask'] * 0.4, 0, 1)
    return blend
