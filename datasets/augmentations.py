import random

import numpy as np
from PIL import Image, ImageEnhance


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def normalize_image(image):
    return (np.asarray(image, dtype=np.float32) - MEAN) / STD


def binary_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    elif mask.ndim == 3 and mask.shape[2] == 3 and np.all(mask == mask[..., :1]):
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("Binary labels must be single-channel images")
    values = set(np.unique(mask).tolist())
    if values <= {0, 1}:
        return mask.astype(np.uint8)
    if values <= {0, 255}:
        return (mask == 255).astype(np.uint8)
    raise ValueError(f"Unsupported label values: {sorted(values)}")


def random_photometric(image, probability=0.5):
    if random.random() > probability:
        return np.asarray(image, dtype=np.float32)
    enhanced = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    for factory in (ImageEnhance.Brightness, ImageEnhance.Contrast,
                    ImageEnhance.Color, ImageEnhance.Sharpness):
        enhanced = factory(enhanced).enhance(1.0 + random.uniform(-0.2, 0.2))
    return np.asarray(enhanced, dtype=np.float32)


def augment_pair(before, after, label, crop_size=256):
    height, width = label.shape
    if before.shape[:2] != (height, width) or after.shape[:2] != (height, width):
        raise ValueError("Image pair and label must have matching dimensions")
    pad_h = max(crop_size - height, 0)
    pad_w = max(crop_size - width, 0)
    if pad_h or pad_w:
        before = np.pad(before, ((0, pad_h), (0, pad_w), (0, 0)))
        after = np.pad(after, ((0, pad_h), (0, pad_w), (0, 0)))
        label = np.pad(label, ((0, pad_h), (0, pad_w)), constant_values=255)
    height, width = label.shape
    top = random.randrange(height - crop_size + 1)
    left = random.randrange(width - crop_size + 1)
    before = before[top:top + crop_size, left:left + crop_size]
    after = after[top:top + crop_size, left:left + crop_size]
    label = label[top:top + crop_size, left:left + crop_size]
    if random.random() > 0.5:
        before, after, label = np.fliplr(before), np.fliplr(after), np.fliplr(label)
    if random.random() > 0.5:
        before, after, label = np.flipud(before), np.flipud(after), np.flipud(label)
    rotations = random.randrange(3) + 1
    before = np.rot90(before, rotations).copy()
    after = np.rot90(after, rotations).copy()
    label = np.rot90(label, rotations).copy()
    return random_photometric(before), random_photometric(after), label
