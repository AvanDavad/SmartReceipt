from src.readers.char_reader import CharReader
from src.readers.image_reader import ImageReader


import tempfile
import unittest
from pathlib import Path


class TestCharReader(unittest.TestCase):
    def setUp(self):
        rootdir = Path(__file__).parent.parent.parent.parent / "test_data" / "train"
        self.image_reader = ImageReader(rootdir)
        self.char_reader = CharReader(self.image_reader)

    def test_reader_len(self):
        assert len(self.char_reader) == 181

    def test_get(self):
        sample = self.char_reader[0]

    def test_show(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.char_reader.show(0, Path(tempdir))

if __name__ == '__main__':
    unittest.main()