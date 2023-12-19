import unittest

import torch
from torch import Tensor
from src.models.phase2char_models import CNNBackbone


class TestCNNBackbone(unittest.TestCase):
    def test_forward(self):
        model = CNNBackbone()
        x = torch.linspace(0, 1, 128 * 128 * 3).reshape(1, 3, 128, 128)
        out: Tensor = model(x)
        assert out.shape == (1, 128, 1, 1)
        assert out.dtype == torch.float32


if __name__ == "__main__":
    unittest.main()
