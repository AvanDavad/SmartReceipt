import tempfile
import unittest
from pathlib import Path

import torch

from src.datasets.phase2char_dataset import Phase2CharDataset
from src.readers.char_reader import CharReader


class TestCharDataset(unittest.TestCase):
    def setUp(self):
        rootdir = Path(__file__).parents[3] / "test_data" / "train"

        self.w = 2
        self.char_reader = CharReader(rootdir, w=self.w)
        self.dataset = Phase2CharDataset(
            self.char_reader,
            augment=False,
            shuffle=False,
        )

    def test_get(self):
        sample_t = self.dataset[0]
        for key, val in sample_t.items():
            assert isinstance(val, torch.Tensor)
        assert sample_t["img"].shape == (
            1 + 2 * self.w,
            3,
            Phase2CharDataset.IMG_SIZE,
            Phase2CharDataset.IMG_SIZE,
        )

    def test_show(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.dataset.show(0, Path(tempdir))


if __name__ == "__main__":
    unittest.main()
