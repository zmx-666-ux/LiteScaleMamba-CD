import unittest

import numpy as np

from datasets.augmentations import binary_mask, normalize_image


class AugmentationTests(unittest.TestCase):
    def test_binary_mask_accepts_zero_one_and_zero_255_encodings(self):
        self.assertTrue(np.array_equal(binary_mask(np.array([[0, 1]], dtype=np.uint8)), [[0, 1]]))
        self.assertTrue(np.array_equal(binary_mask(np.array([[0, 255]], dtype=np.uint8)), [[0, 1]]))

    def test_normalization_uses_imagenet_statistics(self):
        image = np.array([[[123.675, 116.28, 103.53]]], dtype=np.float32)
        self.assertTrue(np.allclose(normalize_image(image), 0.0))


if __name__ == "__main__":
    unittest.main()
