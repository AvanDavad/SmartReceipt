import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

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
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        s = self.reader[idx]

        img = s["img"]
        kps = torch.tensor(s["keypoints"])[:6]
        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = self.transforms(img)

        return img_tensor, kps

if __name__ == "__main__":
    reader = ImageReader("/home/avandavad/projects/receipt_extractor/data/train")
    dataset = ImageDataset(reader)
    inp = dataset[0]