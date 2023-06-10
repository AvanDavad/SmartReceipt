import matplotlib.pyplot as plt
from PIL import Image


import json
from pathlib import Path


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


    def show(self, idx):
        plt.figure(figsize=(16,8))
        s = self[idx]
        plt.imshow(s["img"])
        kps = s["keypoints"]
        for i in range(len(kps)):
            plt.scatter(kps[i][0], kps[i][1], c="b", s=10)
        filename = f"visualization/image_reader/sample_{idx}.jpg"
        plt.savefig(filename)
        print(f"saved {filename}")
        plt.close()