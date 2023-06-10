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

        src_img = np.array(img)
        M = np.array(annot["M"])
        ortho_height = annot["ortho_height"]
        ortho_width = annot["ortho_width"]
        ortho_img = cv2.warpPerspective(src_img, M, (ortho_width, ortho_height))
        ortho_img = Image.fromarray(ortho_img)

        sample = {
            "phase_0": {"img": img, "points": annot["base_points"]},
            "tr_matrix_0_to_1": M,
            "phase_1": {"img": ortho_img, "lines": annot["lines"]},
        }

        return sample


    def show(self, idx, out_folder):
        s = self[idx]
        img = s["phase_0"]["img"]
        base_points = np.array(s["phase_0"]["points"])

        filename = out_folder / f"sample_{idx}.jpg"

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(img)
        ax[0].scatter(base_points[:, 0], base_points[:, 1], s=10, c="r")

        ortho_img = s["phase_1"]["img"]        
        ax[1].imshow(ortho_img)
        for line_y in s["phase_1"]["lines"]:
            ax[1].plot([0, ortho_img.width], [line_y, line_y], c="r", linewidth=1)

        plt.savefig(filename)
        plt.close()
