from src.models.phase0_points.points_model import CNNModulePhase0Points


from PIL import Image
import numpy as np

import unittest
from pathlib import Path

PROJ_DIR = Path(__file__).parents[4]

class TestPhase0PointsModel(unittest.TestCase):
    def setUp(self):
        self.model = CNNModulePhase0Points()
        self.img = Image.open(PROJ_DIR / "test_data" / "train" / "IMG_0001.jpg")

    def test_inference_float(self):
        pred_kps = self.model.inference(self.img)
        assert pred_kps.shape == (4, 2)
        assert pred_kps.dtype == np.float32

    def test_inference_int(self):
        pred_kps = self.model.inference(self.img, as_int=True)
        assert isinstance(pred_kps, np.ndarray)
        assert pred_kps.shape == (4, 2)
        assert pred_kps.dtype == np.int64

    def test_inference_tuple_list(self):
        pred_kps = self.model.inference(self.img, to_tuple_list=True)
        assert len(pred_kps) == 4
        assert isinstance(pred_kps[0], tuple)

if __name__ == "__main__":
    unittest.main()
