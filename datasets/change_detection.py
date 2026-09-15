from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.augmentations import augment_pair, binary_mask, normalize_image


def load_names(list_path):
    names = [line.strip() for line in Path(list_path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names:
        raise ValueError(f"Empty image list: {list_path}")
    if any(Path(name).name != name for name in names):
        raise ValueError("Image lists must contain filenames without directories")
    return names


class ChangeDetectionDataset(Dataset):
    def __init__(self, split_dir, list_path, train=False, crop_size=256):
        self.split_dir = Path(split_dir)
        self.names = load_names(list_path)
        self.train = train
        self.crop_size = crop_size
        for folder in ("A", "B", "label"):
            if not (self.split_dir / folder).is_dir():
                raise FileNotFoundError(self.split_dir / folder)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.split_dir / "A" / name) as image:
            before = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(self.split_dir / "B" / name) as image:
            after = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(self.split_dir / "label" / name) as image:
            label = binary_mask(np.asarray(image))
        if self.train:
            before, after, label = augment_pair(before, after, label, self.crop_size)
        if before.shape[:2] != label.shape or after.shape[:2] != label.shape:
            raise ValueError(f"Image pair and label dimensions differ: {name}")
        before = np.ascontiguousarray(normalize_image(before).transpose(2, 0, 1))
        after = np.ascontiguousarray(normalize_image(after).transpose(2, 0, 1))
        label = np.ascontiguousarray(label.astype(np.int64))
        return torch.from_numpy(before), torch.from_numpy(after), torch.from_numpy(label), name
