from src.readers.image_reader import ImageReader


import tempfile
import unittest
from pathlib import Path


class TestImageReader(unittest.TestCase):
    def setUp(self):
        rootdir = Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        self.reader = ImageReader(rootdir)

    def test_reader_len(self):
        assert len(self.reader) == 1

    def test_get(self):
        _ = self.reader[0]

    def test_show(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.reader.show(0, Path(tempdir))

if __name__ == '__main__':
    unittest.main()