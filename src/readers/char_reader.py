from typing import List, Optional
import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path

import numpy as np
import cv2
from typing import Tuple
from src.readers.image_reader import ImageReader

from dataclasses import dataclass

@dataclass
class PatchSample:
    crop_xyxy: Optional[Tuple[float, float, float, float]] = None
    label: Optional[str] = None
    is_double_space: Optional[bool] = None

@dataclass
class CharSample:
    image: Image.Image
    pre_patches: List[PatchSample]
    patch: PatchSample
    post_patches: List[PatchSample]

    def __repr__(self) -> str:
        self_msg = "CharSample("

        self_msg = self_msg + f"\n\timage {self.image.size}"
        self_msg = self_msg + f"\n\t{len(self.pre_patches)} pre patches"
        self_msg = self_msg + f"\n\t{len(self.post_patches)} post patches"
        self_msg = self_msg + f"\n\tlabel: <{self.patch.label}>"
        self_msg = self_msg + f"\n\tis_double_space: {self.patch.is_double_space}"
        self_msg = self_msg + "\n)"
        return self_msg

class CharReader:
    def __init__(self, image_reader: ImageReader, w: int = 5):
        self.image_reader = image_reader
        self._w = w

        self.mapping = []
        for i in range(len(self.image_reader)):
            sample = self.image_reader[i]
            if sample.phase_2_lines is None:
                continue

            text = sample.phase_2_text
            coords = sample.phase_2_lines
            assert len(text) == len(coords) - 1

            line_width = sample.phase_1_image.width

            num_chars = len(text)
            for j in range(num_chars):
                line_idx_start = int(np.floor(coords[j] / line_width))
                line_idx_end = int(np.floor(coords[j+1] / line_width))
                if line_idx_start == line_idx_end:
                    self.mapping.append((i, j, line_idx_start))

    def __repr__(self):
        return f"CharReader(len={len(self)})"

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx: int) -> CharSample:
        assert idx < len(self)
        assert idx >= 0

        i, j, line_idx = self.mapping[idx]

        i_sample = self.image_reader[i]

        text = i_sample.phase_2_text
        coords = i_sample.phase_2_lines

        img = i_sample.phase_1_image
        lines_y = i_sample.phase_1_lines

        pre_patches: List[PatchSample] = []
        for jj in range(self._w):
            if idx - jj - 1 >= 0:
                pre_i, pre_j, pre_line_idx = self.mapping[idx - jj - 1]
                if i == pre_i and line_idx == pre_line_idx:
                    x0_pre = np.remainder(coords[pre_j], img.width)
                    x1_pre = x0_pre + coords[pre_j+1] - coords[pre_j]
                    patch = PatchSample(
                        crop_xyxy=(x0_pre, lines_y[line_idx], x1_pre, lines_y[line_idx+1]),
                        label=text[pre_j],
                    )
                else:
                    patch = PatchSample()
            else:
                patch = PatchSample()
            pre_patches.insert(0, patch)
        assert len(pre_patches) == self._w

        post_patches: List[PatchSample] = []
        for jj in range(self._w):
            if idx + jj + 1 < len(self):
                post_i, post_j, post_line_idx = self.mapping[idx + jj + 1]
                if i == post_i and line_idx == post_line_idx:
                    x0_post = np.remainder(coords[post_j], img.width)
                    x1_post = x0_post + coords[post_j+1] - coords[post_j]
                    patch = PatchSample(
                        crop_xyxy=(x0_post, lines_y[line_idx], x1_post, lines_y[line_idx+1]),
                        label=text[post_j],
                    )
                else:
                    patch = PatchSample()
            else:
                patch = PatchSample()
            post_patches.append(patch)
        assert len(post_patches) == self._w

        x0 = np.remainder(coords[j], img.width)
        x1 = x0 + coords[j+1] - coords[j]
        is_double_space = (text[j] == " " and post_patches[0].label == " ") or post_patches[0].label is None

        patch = PatchSample(
            crop_xyxy=(x0, lines_y[line_idx], x1, lines_y[line_idx+1]),
            label=text[j],
            is_double_space=is_double_space,
        )

        sample = CharSample(
            image=i_sample.phase_1_image,
            pre_patches=pre_patches,
            patch=patch,
            post_patches=post_patches,
        )

        return sample

    @property
    def window_size(self) -> int:
        return 2*self._w + 1

    def show(self, idx: int, out_folder: Path):
        out_folder.mkdir(parents=True, exist_ok=True)

        sample = self[idx]
        img = sample.image

        filename = out_folder / f"char_sample_{idx}.jpg"

        crop_xyxy = []
        for patch in sample.pre_patches:
            if patch.crop_xyxy is not None:
                crop_xyxy.append(patch.crop_xyxy)
        if sample.patch.crop_xyxy is not None:
            crop_xyxy.append(sample.patch.crop_xyxy)
        for patch in sample.post_patches:
            if patch.crop_xyxy is not None:
                crop_xyxy.append(patch.crop_xyxy)
        crop_xyxy = np.array(crop_xyxy)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img)
        patches_all = sample.pre_patches + [sample.patch] + sample.post_patches
        current_color = "r"
        colors = ["k"] * len(sample.pre_patches) + [current_color] + ["y"] * len(sample.post_patches)
        for patch, color in zip(patches_all, colors):
            if patch.crop_xyxy is not None:
                x0, y0, x1, y1 = patch.crop_xyxy
                ax.plot([x0, x1-1.0, x1-1.0, x0, x0], [y0, y0, y1, y1, y0], c=color, linewidth=2)
                if patch.label is not None and color == current_color:
                    ax.text(
                        (x0+x1)/2,
                        (y0+y1)/2,
                        patch.label,
                        fontsize=28,
                        color=color,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                if patch.is_double_space:
                    ax.plot([x0, x1-1.0], [y0 + 0.1 * (y1-y0), y0 + 0.1 * (y1-y0)], c=color, linewidth=2)
        x0 = crop_xyxy[:, 0].min()
        y0 = crop_xyxy[:, 1].min()
        x1 = crop_xyxy[:, 2].max()
        y1 = crop_xyxy[:, 3].max()

        ax.set_xlim(x0 - (x1-x0) * 0.5, x1 + (x1-x0) * 0.5)
        ax.set_ylim(y1 + (y1-y0) * 0.5, y0 - (y1-y0) * 0.5)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(filename)
        plt.close()
