from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from src.draw_utils import save_img_with_kps

class ImageDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMG_SIZE = 224
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(self, reader, augment):
        self.reader = reader
        self.augment = augment

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        s = self.reader[idx]

        img = s["img"]
        kps = torch.tensor(s["keypoints"])[:6]

        # augment
        if self.augment:
            # rotation augmentation
            if np.random.rand() > 0.5:
                img, kps = ImageDataset.rotate(img, kps)

            img, kps = ImageDataset.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = ImageDataset.TRANSFORMS(img)

        return img_tensor, kps

    @staticmethod
    def rotate(img, kps):
        rotation_angle_deg = np.random.rand() * 30 - 15
        rotation_angle_rad = np.deg2rad(rotation_angle_deg)
        rotation_matrix = np.array([
                        [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                        [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)],
                ])
        rot_torch = torch.from_numpy(rotation_matrix.astype(np.float32))
        img = transforms.functional.rotate(img, np.rad2deg(rotation_angle_rad))

        center = torch.tensor([img.width, img.height]) / 2
        kps = kps - center
        kps = torch.matmul(kps, rot_torch)
        kps = kps + center
        return img,kps

    @staticmethod
    def crop_augment(img, kps):
        kps_x0 = kps[:, 0].min().item()
        kps_x1 = kps[:, 0].max().item()
        kps_y0 = kps[:, 1].min().item()
        kps_y1 = kps[:, 1].max().item()

        crop_x0 = int(kps_x0 * np.random.rand())
        crop_x1 = int(kps_x1 + np.random.rand() * (img.width - kps_x1))
        crop_y0 = int(kps_y0 * np.random.rand())
        crop_y1 = int(kps_y1 + np.random.rand() * (img.height - kps_y1))

        # make square
        crop_1 = max(crop_x1 - crop_x0, crop_y1 - crop_y0)
        crop_y1 = crop_y0 + crop_1
        crop_x1 = crop_x0 + crop_1

        img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        kps = kps - torch.tensor([crop_x0, crop_y0])

        return img, kps

    @staticmethod
    def img_from_tensor(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * np.array(ImageDataset.STD) + np.array(ImageDataset.MEAN)) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def show(self, idx, out_folder, repeat_idx=0):
        img_tensor, kps_tensor = self[idx]

        img = ImageDataset.img_from_tensor(img_tensor)
        kps = kps_tensor.reshape(-1, 2).numpy() * ImageDataset.IMG_SIZE
        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        save_img_with_kps(img, kps, filename, circle_radius=1)
