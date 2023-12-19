from pathlib import Path
from typing import Dict, List
from src.draw_utils import draw_text_on_image, draw_vertical_line, put_stuffs_on_img

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.readers.char_reader import CharReader

def make_square_by_padding(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img

    if w > h:
        img_new = Image.new(img.mode, (w, w), color=(255, 255, 255))
        img_new.paste(img, (0, (w - h) // 2))
    else:
        img_new = Image.new(img.mode, (h, h), color=(255, 255, 255))
        img_new.paste(img, ((h - w) // 2, 0))

    return img_new


class Phase2CharDataset(Dataset):
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

    def __init__(self, reader: CharReader, augment: bool = False, shuffle: bool = False):
        assert isinstance(reader, CharReader)
        self.reader = reader
        self.augment = augment

        self.idx_list = (
            np.random.permutation(len(self.reader))
            if shuffle
            else np.arange(len(self.reader))
        )

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        idx_mod = self.idx_list[idx]
        sample = self.reader[idx_mod]

        patches = sample.pre_patches + [sample.patch] + sample.post_patches

        img_patches = []
        for patch_idx, patch in enumerate(patches):
            if patch.crop_xyxy is not None:
                x0, y0, x1, y1 = patch.crop_xyxy
                if y1 - y0 > x1 - x0:
                    img_sidelen = int(y1 - y0)
                else:
                    img_sidelen = int(x1 - x0)

                y_offset = 0.0 if not self.augment else img_sidelen * np.random.uniform(-0.1, 0.1)

                x_left = int(x0)
                y_top = int(y0 + (y1 - y0)/2 - img_sidelen/2 + y_offset)
                x_right = int(x_left + img_sidelen)
                y_bottom = int(y_top + img_sidelen)

                img_patch = sample.image.crop((x_left, y_top, x_right, y_bottom))
                if patch_idx == len(sample.pre_patches):
                    char_width = (x1 - x0) / img_sidelen
            else:
                img_patch = Image.new(
                    "RGB",
                    (Phase2CharDataset.IMG_SIZE, Phase2CharDataset.IMG_SIZE),
                    color=(0, 0, 0),
                )
            img_patches.append(
                Phase2CharDataset.TRANSFORMS(img_patch)
            )

        sample_t = {
            "img":torch.stack(img_patches, dim=0),
            "label":torch.tensor([ord(sample.patch.label)]),
            "is_double_space":torch.tensor([sample.patch.is_double_space]),
            "char_width": torch.tensor([char_width]),
        }

        return sample_t

    @staticmethod
    def img_from_tensor(img_tensor: Tensor) -> Image.Image:
        img: np.ndarray = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase2CharDataset.STD) + np.array(Phase2CharDataset.MEAN)
        ) * 255.0
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)
        return img_pil

    def show(self, idx: int, out_folder: Path, repeat_idx: int=0, verbose: bool = False):
        sample_t = self[idx]

        images: List[Image.Image] = []
        for img_idx, img_tensor in enumerate(sample_t["img"]):
            img = Phase2CharDataset.img_from_tensor(img_tensor)
            if img_idx == self.reader._w:
                label_chr = chr(sample_t["label"].item())
                img = draw_text_on_image(img, text=f"<{label_chr}>", pos=(0,0))
                char_width = sample_t["char_width"].item()
                img = draw_vertical_line(img, int(img.width * char_width))

            images.append(img)

        pad_size = 10
        new_height = self.IMG_SIZE + 2*pad_size
        new_width = (self.IMG_SIZE + pad_size) * len(images) + pad_size
        img_big = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

        for img_idx, img in enumerate(images):
            img_big.paste(img, (pad_size + img_idx*(self.IMG_SIZE + pad_size), pad_size))

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        img_big.save(filename)
        if verbose:
            print(f"Saved {filename}")
