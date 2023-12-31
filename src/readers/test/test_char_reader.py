from src.readers.char_reader import CharReader
from src.readers.image_reader import ImageReader


import tempfile
import unittest
from pathlib import Path
import numpy as np

class TestCharReader(unittest.TestCase):
    def setUp(self):
        rootdir = Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        self.image_reader = ImageReader(rootdir)
        self.char_reader = CharReader(self.image_reader)

    def test_reader_len(self):
        assert len(self.char_reader) == 177

    def test_get(self):
        self.char_reader[0]

    def test_show(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.char_reader.show(0, Path(tempdir))

    def test_patches_coords_and_Nones(self):
        for i in range(len(self.char_reader)):
            sample = self.char_reader[i]

            crop_last = None
            for patch in sample.pre_patches:
                if patch.crop_xyxy is not None:
                    x0, y0, x1, y1 = patch.crop_xyxy
                    assert x0 < x1
                    assert y0 < y1
                    if crop_last is not None:
                        np.testing.assert_almost_equal(
                            crop_last[2], x0, decimal=3,
                        )
                        np.testing.assert_almost_equal(
                            crop_last[1], y0, decimal=3,
                        )
                        np.testing.assert_almost_equal(
                            crop_last[3], y1, decimal=3,
                        )
                    crop_last = patch.crop_xyxy
                else:
                    assert crop_last is None

            if crop_last is not None:
                np.testing.assert_almost_equal(
                    crop_last[2], sample.patch.crop_xyxy[0], decimal=3,
                )
                np.testing.assert_almost_equal(
                    crop_last[1], sample.patch.crop_xyxy[1], decimal=3,
                )
                np.testing.assert_almost_equal(
                    crop_last[3], sample.patch.crop_xyxy[3], decimal=3,
                )

            crop_last = sample.patch.crop_xyxy

            for patch in sample.post_patches:
                if patch.crop_xyxy is not None:
                    assert crop_last is not None

                    x0, y0, x1, y1 = patch.crop_xyxy
                    assert x0 < x1
                    assert y0 < y1
                    np.testing.assert_almost_equal(
                        crop_last[2], x0, decimal=3,
                    )
                    np.testing.assert_almost_equal(
                        crop_last[1], y0, decimal=3,
                    )
                    np.testing.assert_almost_equal(
                        crop_last[3], y1, decimal=3,
                    )

                crop_last = patch.crop_xyxy

if __name__ == '__main__':
    unittest.main()