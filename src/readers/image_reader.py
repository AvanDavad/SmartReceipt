import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path

import numpy as np
import cv2

from dataclasses import dataclass

from typing import Optional
from typing import Union
from typing import List
from typing import Tuple


@dataclass
class Sample:
    phase_0_image: Image.Image
    phase_0_points: np.ndarray
    phase_1_image: Optional[Image.Image] = None
    phase_1_lines: Optional[np.ndarray] = None
    phase_1_tr_matrix_0_to_1: Optional[np.ndarray] = None
    phase_2_text: Optional[str] = None
    phase_2_lines: Optional[np.ndarray] = None

class ImageReader:
    def __init__(self, rootdir: Union[str, Path]):
        self.rootdir = Path(rootdir)
        self.data: List[Tuple[Path, Path]] = []
        for img_name in self.rootdir.glob("*.jpg"):
            annot_name = img_name.with_suffix(".json")
            if annot_name.is_file():
                self.data.append((img_name, annot_name))

    def __repr__(self):
        return f"ImageReader({len(self)} images)"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        assert idx < len(self)
        assert idx >= 0

        img_name, annot_name = self.data[idx]
        img = Image.open(str(img_name))

        with open(str(annot_name), "r") as file:
            annot = json.load(file)

        # phase 0
        phase_0_points = np.array(annot["base_points"])

        # phase 1
        phase_1_ortho_img = None
        phase_1_lines = None
        phase_1_tr_matrix_0_to_1 = None
        if "M" in annot:
            src_img = np.array(img)
            M = np.array(annot["M"])
            ortho_height = annot["ortho_height"]
            ortho_width = annot["ortho_width"]
            phase_1_ortho_img = cv2.warpPerspective(src_img, M, (ortho_width, ortho_height))
            phase_1_ortho_img = Image.fromarray(phase_1_ortho_img)

            phase_1_lines = np.array(annot["lines_y"])
            phase_1_tr_matrix_0_to_1 = M

        # phase 2 and 3
        phase_2_text = None
        phase_2_lines = None
        if "text" in annot:
            phase_2_text = annot["text"]
            phase_2_lines = np.array(annot["lines_x"])

        sample = Sample(
            phase_0_image=img,
            phase_0_points=phase_0_points,
            phase_1_image=phase_1_ortho_img,
            phase_1_lines=phase_1_lines,
            phase_1_tr_matrix_0_to_1=phase_1_tr_matrix_0_to_1,
            phase_2_text=phase_2_text,
            phase_2_lines=phase_2_lines,
        )

        return sample


    def show(self, idx: int, out_folder: Path):
        out_folder = out_folder / str(idx).zfill(4)
        out_folder.mkdir(parents=True, exist_ok=True)

        sample = self[idx]
        img = sample.phase_0_image
        base_points = np.array(sample.phase_0_points)

        filename = out_folder / "sample_phase_0.jpg"

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.scatter(base_points[:, 0], base_points[:, 1], s=10, c="r")
        plt.savefig(filename)
        plt.close()

        if sample.phase_1_image is not None:
            filename = out_folder / "sample_phase_1.jpg"
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ortho_img = sample.phase_1_image
            ax.imshow(ortho_img)
            for line_y in sample.phase_1_lines:
                ax.plot([0, ortho_img.width], [line_y, line_y], c="r", linewidth=1)
            plt.savefig(filename)
            plt.close()

        if sample.phase_2_text is not None:
            ortho_img_np = np.array(ortho_img)
            out_folder = out_folder / "phase_2"
            out_folder.mkdir(parents=True, exist_ok=True)
            line_y0 = sample.phase_1_lines[:-1]
            line_y1 = sample.phase_1_lines[1:]
            x_offset = 0
            text_offset = 0
            for i, (ly0, ly1) in enumerate(zip(line_y0, line_y1)):
                img_line = ortho_img_np[int(ly0):int(ly1), :, :]
                fig, ax = plt.subplots(3, 1, figsize=(10, 3))
                ax[0].imshow(img_line)
                ax[1].imshow(img_line)
                ax[2].imshow(img_line // 2)
                first_line = True
                for j, x in enumerate(sample.phase_2_lines):
                    if x - x_offset < 0:
                        continue
                    if x - x_offset > img_line.shape[1]:
                        break
                    ax[1].plot([x - x_offset, x - x_offset], [0, img_line.shape[0]], c="r", linewidth=1)
                    ax[2].plot([x - x_offset, x - x_offset], [0, img_line.shape[0]], c="r", linewidth=1)
                    if not first_line:
                        ch = sample.phase_2_text[j-1]
                        x0 = sample.phase_2_lines[j-1] - x_offset
                        x1 = sample.phase_2_lines[j] - x_offset
                        ax[2].text(
                            (x0+x1)/2,
                            img_line.shape[0]/2,
                            ch,
                            fontsize=14,
                            color="white",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )
                    first_line = False
                ax[0].set_xticks([])
                ax[1].set_xticks([])
                ax[2].set_xticks([])
                ax[0].set_yticks([])
                ax[1].set_yticks([])
                ax[2].set_yticks([])

                filename = out_folder / f"sample_phase_2{i}.jpg"
                plt.savefig(filename)
                plt.close()

                x_offset += img_line.shape[1]
                text_offset += j-1
