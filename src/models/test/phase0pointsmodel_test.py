import tempfile
import unittest
from pathlib import Path

from src.models.phase0points_model import CNNModulePhase0Points

PROJ_DIR = Path(__file__).parent.parent.parent.parent


class TestPhase0PointsModel(unittest.TestCase):
    def setUp(self):
        self.model = CNNModulePhase0Points()

    def test_inference(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.model.inference(
                PROJ_DIR / "test_data" / "train" / "IMG_0001.jpg",
                out_folder=Path(tempdir),
            )


if __name__ == "__main__":
    unittest.main()
