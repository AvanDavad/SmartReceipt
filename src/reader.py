from torch.utils.data import Dataset
import torch
import numpy as np
from src.image_dataset import ImageDataset
from src.image_reader import ImageReader



class Top2PointsDataset(Dataset):
    def __init__(self, reader, augment=False, zoom_in_factor=5.0):
        self.reader = reader
        self.augment = augment
        self.zoom_in_factor = zoom_in_factor

    def __len__(self):
        return 2 * len(self.reader)

    def __getitem__(self, idx):
        s = self.reader[idx // 2]

        img = s["img"]
        offset = 2 * (idx % 2)
        kps = torch.tensor(s["keypoints"])[offset : offset + 2]

        img, kps = Top2PointsDataset.zoom_in(img, kps, self.zoom_in_factor)

        # augment
        if self.augment:
            img, kps = ImageDataset.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = ImageDataset.TRANSFORMS(img)

        return img_tensor, kps

    @staticmethod
    def zoom_in(img, kps, zoom_in_factor):
        assert kps.shape == (2, 2)

        len_line = np.linalg.norm(kps[0] - kps[1])
        center = (kps[0] + kps[1]) / 2.0

        crop_x0 = int(center[0] - len_line * zoom_in_factor / 2.0)
        crop_x1 = int(center[0] + len_line * zoom_in_factor / 2.0)
        crop_y0 = int(center[1] - len_line * zoom_in_factor / 2.0)
        crop_y1 = int(center[1] + len_line * zoom_in_factor / 2.0)

        img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        kps = kps - torch.tensor([crop_x0, crop_y0])

        return img, kps
