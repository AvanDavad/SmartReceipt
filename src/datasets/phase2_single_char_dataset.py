from pathlib import Path
from typing import Dict, List
from src.datasets.phase2char_dataset import ALL_CHARS
from src.draw_utils import draw_text_on_image, draw_vertical_line

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.readers.char_reader import CharReader




class Phase2SingleCharDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.2, 0.2, 0.2]
    IMG_SIZE = 128
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(
        self, char_reader: CharReader, augment: bool = False, shuffle: bool = False
    ):
        assert isinstance(char_reader, CharReader)
        self.char_reader = char_reader
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
        if y1 - y0 > x1 - x0:
            img_sidelen = int(y1 - y0)
        else:
            img_sidelen = int(x1 - x0)

        if self.augment:
            img_sidelen = int(img_sidelen * np.random.uniform(0.8, 1.4))

        img_patch = sample.image.crop((x0, y0, x1, y1))
        img_square = Image.new("RGB", (img_sidelen, img_sidelen), color=(255, 255, 255))
        x_left = int((img_sidelen - img_patch.width) / 2)
        y_top = int((img_sidelen - img_patch.height) / 2)
        if self.augment:
            x_left += int(img_sidelen * np.random.uniform(-0.2, 0.2))
            y_top += int(img_sidelen * np.random.uniform(-0.2, 0.2))
        img_square.paste(img_patch, (x_left, y_top))

        if sample.patch.label in ALL_CHARS:
            target = ALL_CHARS.index(sample.patch.label)
        else:
            target = len(ALL_CHARS)

        sample_t = {
            "img": Phase2SingleCharDataset.TRANSFORMS(img_square),
            "target": torch.tensor([target]),
        }

        return sample_t

    @staticmethod
    def img_from_tensor(img_tensor: Tensor) -> Image.Image:
        img: np.ndarray = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase2SingleCharDataset.STD) + np.array(Phase2SingleCharDataset.MEAN)
        ) * 255.0
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)
        return img_pil

    def show(
        self, idx: int, out_folder: Path, repeat_idx: int = 0, verbose: bool = False
    ):
        sample_t = self[idx]

        target_idx = sample_t["target"].item()
        if target_idx < len(ALL_CHARS):
            label_chr = ALL_CHARS[target_idx]
            if label_chr == " ":
                label_chr = "space"
        else:
            label_chr = "@?@"

        img = Phase2SingleCharDataset.img_from_tensor(sample_t["img"])
        img = draw_text_on_image(img, label_chr, pos=(0, 0))

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        img.save(filename)
        if verbose:
            print(f"Saved {filename}")
