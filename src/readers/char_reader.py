from typing import List, Optional
import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path

import numpy as np
from typing import Tuple

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
    def __init__(self, root_dir: Path, w: int = 5):
        self.root_dir = root_dir / "char_reader"
        assert self.root_dir.is_dir()

        self._w = w

        self.filenames = sorted(self.root_dir.glob("*.jpg"))
        self.images: List[Image.Image] = []

        self.mapping: List[Tuple[int, float, float, str]] = []
        for i, filename in enumerate(self.filenames):
            image = Image.open(filename)
            self.images.append(image.copy())

            json_filename = filename.with_suffix(".json")
            assert json_filename.is_file()
            with open(json_filename, "r") as f:
                data = json.load(f)
            line_coords = np.array(data["line_coords"])
            line_text = data["line_text"]
            for j, (x0, x1) in enumerate(zip(line_coords[:-1], line_coords[1:])):
                self.mapping.append((i, x0, x1, line_text[j]))

    def __repr__(self):
        return f"CharReader(len={len(self)})"

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx: int) -> CharSample:
        assert idx < len(self)
        assert idx >= 0

        current_line_idx, x0, x1, char = self.mapping[idx]

        img = self.images[current_line_idx]
        height_per_3 = img.height // 3

        this_char_is_space = char == " "
        next_char_is_like_space = True
        if idx + 1 < len(self):
            line_idx, _, _, char_next = self.mapping[idx + 1]
            if current_line_idx == line_idx and char_next != " ":
                next_char_is_like_space = False
        is_double_space = this_char_is_space and next_char_is_like_space

        current_patch = PatchSample(
            crop_xyxy=(x0, height_per_3, x1, 2*height_per_3),
            label=char,
            is_double_space=is_double_space,
        )

        pre_patches: List[PatchSample] = []
        for jj in range(self._w):
            if idx - jj - 1 >= 0:
                line_idx, x0_pre, x1_pre, char = self.mapping[idx - jj - 1]

                if current_line_idx == line_idx:
                    patch = PatchSample(
                        crop_xyxy=(x0_pre, height_per_3, x1_pre, 2*height_per_3),
                        label=char,
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
                line_idx, x0_post, x1_post, char = self.mapping[idx + jj + 1]

                if current_line_idx == line_idx:
                    patch = PatchSample(
                        crop_xyxy=(x0_post, height_per_3, x1_post, 2*height_per_3),
                        label=char,
                    )
                else:
                    patch = PatchSample()
            else:
                patch = PatchSample()
            post_patches.append(patch)
        assert len(post_patches) == self._w

        sample = CharSample(
            image=self.images[current_line_idx],
            pre_patches=pre_patches,
            patch=current_patch,
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
                        fontsize=14,
                        color=color,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                if patch.is_double_space:
                    ax.plot([x0, x1-1.0], [y0 + 0.1 * (y1-y0), y0 + 0.1 * (y1-y0)], c=color, linewidth=1)
        x0 = crop_xyxy[:, 0].min()
        y0 = crop_xyxy[:, 1].min()
        x1 = crop_xyxy[:, 2].max()
        y1 = crop_xyxy[:, 3].max()

        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(filename)
        plt.close()
