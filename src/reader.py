import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
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

        # augment
        img, kps = self.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = self.transforms(img)

        return img_tensor, kps

    def crop_augment(self, img, kps):
        kps_x0 = kps[:,0].min().item()
        kps_x1 = kps[:,0].max().item()
        kps_y0 = kps[:,1].min().item()
        kps_y1 = kps[:,1].max().item()

        crop_x0 = int(kps_x0 * np.random.rand())
        crop_x1 = int(kps_x1 + np.random.rand() * (img.width - kps_x1))
        crop_y0 = int(kps_y0 * np.random.rand())
        crop_y1 = int(kps_y1 + np.random.rand() * (img.height - kps_y1))

        # make square
        crop_1 = max(crop_x1-crop_x0, crop_y1-crop_y0)
        crop_y1 = crop_y0 + crop_1
        crop_x1 = crop_x0 + crop_1

        img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        kps = kps - torch.tensor([crop_x0, crop_y0])

        return img, kps

if __name__ == "__main__":
    reader = ImageReader("/home/avandavad/projects/receipt_extractor/data/train")
    dataset = ImageDataset(reader)
    inp = dataset[0]