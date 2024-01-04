from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.draw_utils import put_stuffs_on_img
from src.readers.image_reader import ImageReader


class Phase1LineDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.2, 0.2, 0.2]
    IMG_SIZE = 100
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(self, image_reader: ImageReader, augment: bool=False, shuffle: bool=False):
        self.image_reader = image_reader
        self.augment = augment

        self.mapping: List[Tuple[int, int, int]] = []
        for i in range(len(self.image_reader)):
            s = self.image_reader[i]
            num_lines = s.phase_1_lines.shape[0] - 1
            for j in range(num_lines):
                self.mapping.append((i, j, num_lines))

        self.indices: np.ndarray = (
            np.random.permutation(len(self.mapping))
            if shuffle
            else np.arange(len(self.mapping))
        )

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx: int):
        idx_mod = self.indices[idx]
        sample_idx, line_idx, _ = self.mapping[idx_mod]
        sample = self.image_reader[sample_idx]

        img = sample.phase_1_image
        lines = sample.phase_1_lines

        y_offset_min = (
            (lines[line_idx] if line_idx==0 else (lines[line_idx - 1] + lines[line_idx] * 2) // 3)
            if self.augment
            else lines[line_idx]
        )
        y_offset_max = (
            (lines[line_idx] * 2 + lines[line_idx + 1]) // 3
            if self.augment
            else lines[line_idx]
        )
        y_offset = np.random.randint(y_offset_min, y_offset_max + 1)

        quarter_size = img.width // 4
        img_0 = img.crop((0, y_offset, quarter_size, y_offset + quarter_size))
        img_1 = img.crop((quarter_size, y_offset, quarter_size * 2, y_offset + quarter_size))
        img_2 = img.crop((quarter_size * 2, y_offset, quarter_size * 3, y_offset + quarter_size))
        img_3 = img.crop((quarter_size * 3, y_offset, quarter_size * 4, y_offset + quarter_size))

        img_0_tensor = Phase1LineDataset.TRANSFORMS(img_0)
        img_1_tensor = Phase1LineDataset.TRANSFORMS(img_1)
        img_2_tensor = Phase1LineDataset.TRANSFORMS(img_2)
        img_3_tensor = Phase1LineDataset.TRANSFORMS(img_3)

        input_tensor = torch.cat([img_0_tensor, img_1_tensor, img_2_tensor, img_3_tensor], dim=0)
        line_y = torch.tensor([(lines[line_idx + 1] - y_offset) / quarter_size]).float()

        sample_t = {
            "input": input_tensor,
            "line_y": line_y,
        }

        return sample_t

    @staticmethod
    def images_from_tensor(input_tensor: torch.Tensor) -> List[Image.Image]:
        images = []
        for i in range(input_tensor.shape[0] // 3):
            img = input_tensor[3*i:3*(i+1)].permute(1, 2, 0).numpy()
            img = (
                img * np.array(Phase1LineDataset.STD)
                + np.array(Phase1LineDataset.MEAN)
            ) * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            images.append(img)
        return images

    def show(self, idx: int) -> Image.Image:
        sample_t = self[idx]

        images = Phase1LineDataset.images_from_tensor(sample_t["input"])
        line_y = int(sample_t["line_y"].item() * images[0].width)

        images_with_lines = []
        for img in images:
            line: Tuple[int, int, int, int] = (0, line_y, img.width, line_y)
            img = put_stuffs_on_img(
                img, lines=[line], lines_colors="red", lines_width=2
            )
            images_with_lines.append(img)

        # compose images_with_lines
        margin = 10
        composite_img = Image.new(
            "RGB",
            (
                images_with_lines[0].width * len(images_with_lines) + margin * (len(images_with_lines) + 1),
                images_with_lines[0].height + margin * 2,
            ),
        )
        for i, img_with_line in enumerate(images_with_lines):
            composite_img.paste(img_with_line, (margin + (img_with_line.width + margin) * i, margin))

        return composite_img
