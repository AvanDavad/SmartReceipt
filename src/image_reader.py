import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path

from src.draw_utils import save_img_with_kps
import numpy as np

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

        sample = {
            "img": img,
            "keypoints": annot["base_points"],
        }
        if "lines" in annot:
            sample["lines"] = annot["lines"]

        return sample


    def show(self, idx, out_folder):
        s = self[idx]
        img = s["img"]
        kps = np.array(s["keypoints"])

        filename = out_folder / f"sample_{idx}.jpg"

        save_img_with_kps(img, kps, filename, circle_color="yellow", circle_radius=7)
