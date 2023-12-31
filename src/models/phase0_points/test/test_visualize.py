import unittest

from PIL import Image

from src.models.phase0_points.visualize import visualize_phase0_points


class TestPhase0Visualization(unittest.TestCase):
    def test_visualize(self):
        img = Image.new("RGB", (400, 400), color="white")
        pred_kps = [(0, 0), (100, 100), (200, 200), (300, 300)]

        new_img = visualize_phase0_points(img, pred_kps)

        assert isinstance(new_img, Image.Image)
        assert new_img.size == (400, 400)


if __name__ == "__main__":
    unittest.main()
