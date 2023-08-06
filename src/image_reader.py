import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path

from src.draw_utils import save_img_with_kps
import numpy as np
import cv2

class ImageReader:
    def __init__(self, rootdir):
        self.rootdir = Path(rootdir)
        self.data = []
        for img_name in self.rootdir.glob("*.jpg"):
            annot_name = img_name.with_suffix(".json")
            if annot_name.is_file():
                self.data.append((img_name, annot_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(str(self.data[idx][0]))

        with open(str(self.data[idx][1]), "r") as file:
            annot = json.load(file)

        # phase 0
        sample = {"phase_0": {}}
        sample["phase_0"]["img"] = img
        sample["phase_0"]["points"] = annot["base_points"]

        # phase 1
        if "M" in annot:
            src_img = np.array(img)
            M = np.array(annot["M"])
            ortho_height = annot["ortho_height"]
            ortho_width = annot["ortho_width"]
            ortho_img = cv2.warpPerspective(src_img, M, (ortho_width, ortho_height))
            ortho_img = Image.fromarray(ortho_img)

            sample["phase_1"] = {
                "img": ortho_img,
                "lines": annot["lines_y"],
                "tr_matrix_0_to_1": M,
            }

        # phase 2 and 3
        if "text" in annot:
            sample["phase_2_3"] = {
                "text": annot["text"],
                "lines": annot["lines_x"],
            }

        return sample


    def show(self, idx, out_folder):
        out_folder = out_folder / str(idx).zfill(4)
        out_folder.mkdir(parents=True, exist_ok=True)

        s = self[idx]
        img = s["phase_0"]["img"]
        base_points = np.array(s["phase_0"]["points"])

        filename = out_folder / "sample_phase_0.jpg"

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.scatter(base_points[:, 0], base_points[:, 1], s=10, c="r")
        plt.savefig(filename)
        plt.close()

        if "phase_1" in s:
            filename = out_folder / "sample_phase_1.jpg"
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ortho_img = s["phase_1"]["img"]
            ax.imshow(ortho_img)
            for line_y in s["phase_1"]["lines"]:
                ax.plot([0, ortho_img.width], [line_y, line_y], c="r", linewidth=1)
            plt.savefig(filename)
            plt.close()

        if "phase_2_3" in s:
            ortho_img_np = np.array(ortho_img)
            out_folder = out_folder / "phase_2_3"
            out_folder.mkdir(parents=True, exist_ok=True)
            line_y0 = s["phase_1"]["lines"][:-1]
            line_y1 = s["phase_1"]["lines"][1:]
            x_offset = 0
            for i, (ly0, ly1) in enumerate(zip(line_y0, line_y1)):
                img_line = ortho_img_np[int(ly0):int(ly1), :, :]
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].imshow(img_line)
                ax[1].imshow(img_line)
                for x in s["phase_2_3"]["lines"]:
                    if x - x_offset < 0:
                        continue
                    if x - x_offset > img_line.shape[1]:
                        break
                    ax[1].plot([x - x_offset, x - x_offset], [0, img_line.shape[0]], c="r", linewidth=1)

                filename = out_folder / f"sample_phase_2_3_{i}.jpg"
                plt.savefig(filename)
                plt.close()

                x_offset += img_line.shape[1]
