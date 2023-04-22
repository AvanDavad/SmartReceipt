import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.data import Dataset

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
        img = cv2.imread(str(self.data[idx][0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with open(str(self.data[idx][1]), "r") as file:
            annot = json.load(file)
        sample = {
            "img": img,
            "keypoints": annot,
        }
        return sample
    
    def show(self, idx):
        plt.figure(figsize=(16,8))
        s = self[idx]
        plt.imshow(s["img"])
        kps = s["keypoints"]
        for i in range(12):
            plt.scatter(kps[i][0], kps[i][1], c="b", s=10)

class ImageDataset(Dataset):
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        s = self.reader[idx]