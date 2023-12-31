from pathlib import Path
from typing import Dict
from src.draw_utils import draw_text_on_image, draw_vertical_line

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.readers.char_reader import CharReader



class Phase2CharBorderDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.2, 0.2, 0.2]
    IMG_SIZE = 64
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(
        self, reader: CharReader, augment: bool = False, shuffle: bool = False
    ):
        assert isinstance(reader, CharReader)
        self.char_reader = reader
        self.augment = augment

        self.idx_list = (
            np.random.permutation(len(self.char_reader))
            if shuffle
            else np.arange(len(self.char_reader))
        )

    def __len__(self):
        return len(self.char_reader)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        idx_mod = self.idx_list[idx]
        sample = self.char_reader[idx_mod]

        x0, y0, x1, y1 = sample.patch.crop_xyxy
        img_crop_size = max(x1 - x0, y1 - y0)
        if self.augment:
            img_crop_size *= np.random.uniform(0.8, 1.3)

        x_left = x0
        y_top = y0 + (y1 - y0) / 2 - img_crop_size / 2
        if self.augment:
            x_left += np.random.uniform(-0.1*(x1-x0), 0.1*(x1-x0))
            y_top += np.random.uniform(-0.1, 0.1) * (y1-y0)

        img_crop = sample.image.crop((int(x_left), int(y_top), int(x_left) + img_crop_size, int(y_top) + img_crop_size))
        assert img_crop.width == img_crop.height

        img_tensor = Phase2CharBorderDataset.TRANSFORMS(img_crop)

        sample_t = {
            "img": img_tensor,
            "is_double_space": torch.tensor([sample.patch.is_double_space]),
            "char_end_x": torch.tensor([(x1 - x_left) / img_crop_size]),
        }

        return sample_t

    @staticmethod
    def img_from_tensor(img_tensor: Tensor) -> Image.Image:
        img: np.ndarray = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase2CharBorderDataset.STD) + np.array(Phase2CharBorderDataset.MEAN)
        ) * 255.0
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)
        return img_pil

    def show(
        self, idx: int, out_folder: Path, repeat_idx: int = 0, verbose: bool = False
    ):
        sample_t = self[idx]

        img_tensor: Tensor = sample_t["img"]
        img = Phase2CharBorderDataset.img_from_tensor(img_tensor)

        char_end_x = sample_t["char_end_x"].item()
        img = draw_vertical_line(img, int(img.width * char_end_x), width=2)
        is_double_space = sample_t["is_double_space"].item()
        if is_double_space:
            img = draw_text_on_image(img, text="d.s.", pos=(0, img.height // 2))

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        img.save(filename)
        if verbose:
            print(f"Saved {filename}")
