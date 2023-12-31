from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.readers.image_reader import ImageReader

import torch
import tempfile
import unittest
from pathlib import Path
from PIL import Image


class TestPhase0PointsDataset(unittest.TestCase):
    def setUp(self):
        rootdir = (
            Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        )
        self.image_reader = ImageReader(rootdir)

    def test_dataset_len(self):
        dataset = Phase0PointsDataset(self.image_reader, augment=False)
        assert len(dataset) == 1

    def test_get(self):
        dataset = Phase0PointsDataset(self.image_reader, augment=False)
        sample_t = dataset[0]
        for key, val in sample_t.items():
            assert isinstance(val, torch.Tensor)

    def test_show(self):
        dataset = Phase0PointsDataset(self.image_reader, augment=False)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset.show(0, Path(tempdir))


class TestPhase0PointsDatasetAugment(unittest.TestCase):
    def setUp(self):
        rootdir = (
            Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        )
        image_reader = ImageReader(rootdir)

        sample = image_reader[0]

        self.img = sample.phase_0_image
        self.kps = torch.tensor(sample.phase_0_points).to(dtype=torch.float32)

    def test_crop_augment(self):
        img, kps = Phase0PointsDataset.color_augment(self.img, self.kps)
        assert isinstance(img, Image.Image)
        assert isinstance(kps, torch.Tensor)

    def test_rotate_augment(self):
        img, kps = Phase0PointsDataset.rotate(self.img, self.kps)
        assert isinstance(img, Image.Image)
        assert isinstance(kps, torch.Tensor)

    def test_perspective_augment(self):
        img, kps = Phase0PointsDataset.perspective_augment(self.img, self.kps)
        assert isinstance(img, Image.Image)
        assert isinstance(kps, torch.Tensor)

    def test_crop_augment(self):
        img, kps = Phase0PointsDataset.crop_augment(self.img, self.kps)
        assert isinstance(img, Image.Image)
        assert isinstance(kps, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
