from pathlib import Path
import tempfile
import unittest
from src.datasets.phase0points_dataset import Phase0PointsDataset

from src.readers.image_reader import ImageReader

class TestPhase0PointsDataset(unittest.TestCase):
    def setUp(self):
        rootdir = Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        self.reader = ImageReader(rootdir)

    def test_dataset_len(self):
        dataset = Phase0PointsDataset(self.reader, augment=False)
        assert len(dataset) == 1

    def test_get(self):
        dataset = Phase0PointsDataset(self.reader, augment=False)
        img_tensor, kps = dataset[0]

    def test_augment(self):
        dataset = Phase0PointsDataset(self.reader, augment=True)
        img_tensor, kps = dataset[0]

    def test_show(self):
        dataset = Phase0PointsDataset(self.reader, augment=False)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset.show(0, Path(tempdir))

if __name__ == '__main__':
    unittest.main()