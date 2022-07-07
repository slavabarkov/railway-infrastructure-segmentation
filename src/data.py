from typing import Optional, List, Dict, Any

from pathlib import Path
import numpy as np
import torch
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split, KFold


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: Optional[List[Path]] = None,
            transforms: Optional[A.core.composition.Compose] = None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        image /= 255.

        result = {"image": image}

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(str(mask_path))
            mask[:, :, 0] = mask[:, :, 0] == 6
            mask[:, :, 1] = mask[:, :, 1] == 7
            mask[:, :, 2] = mask[:, :, 2] == 10
            mask = mask.astype('float32')
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)
            if self.masks is not None:
                result["mask"] = result["mask"].permute(2, 0, 1)

        result["filename"] = image_path.name

        return result


def get_loaders(
        images: List[Path],
        masks: List[Path],
        random_state: int,
        valid_size: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 2,
        train_transforms_fn: Optional[A.core.composition.Compose] = None,
        valid_transforms_fn: Optional[A.core.composition.Compose] = None,
        k_folds: int = 4,
        current_fold: int = 0
) -> Dict[str, torch.utils.data.DataLoader]:
    indices = np.arange(len(images))

    skf = KFold(k_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(indices))
    id_train, id_test = splits[current_fold]
    id_test, id_val = train_test_split(id_test, test_size=valid_size, random_state=random_state)

    np_images = np.array(images)
    np_masks = np.array(masks)

    train_dataset = SegmentationDataset(
        images=np_images[id_train].tolist(),
        masks=np_masks[id_train].tolist(),
        transforms=train_transforms_fn
    )

    val_dataset = SegmentationDataset(
        images=np_images[id_val].tolist(),
        masks=np_masks[id_val].tolist(),
        transforms=valid_transforms_fn
    )

    test_dataset = SegmentationDataset(
        images=np_images[id_test].tolist(),
        masks=np_masks[id_test].tolist(),
        transforms=valid_transforms_fn
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    return loaders