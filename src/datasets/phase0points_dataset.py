from pathlib import Path
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.draw_utils import save_img_with_kps
from src.readers.image_reader import ImageReader


class Phase0PointsDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.2, 0.2, 0.2]
    IMG_SIZE = 768
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    def __init__(self, reader: ImageReader, augment: bool = False):
        assert isinstance(reader, ImageReader)
        self.reader = reader
        self.augment = augment

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.reader[idx]

        img = sample.phase_0_image
        kps = torch.tensor(sample.phase_0_points).to(dtype=torch.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img, kps = Phase0PointsDataset.color_augment(img, kps)

            if np.random.rand() < 0.5:
                img, kps = Phase0PointsDataset.rotate(img, kps)

            if np.random.rand() < 0.5:
                img, kps = Phase0PointsDataset.perspective_augment(img, kps)

            if np.random.rand() < 0.5:
                img, kps = Phase0PointsDataset.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = Phase0PointsDataset.TRANSFORMS(img)

        sample_t = {
            "img": img_tensor,
            "kps": kps,
        }

        return sample_t

    @staticmethod
    def color_augment(
        img: Image.Image, kps: Tensor
    ) -> Tuple[Image.Image, Tensor]:
        img = TF.adjust_brightness(img, 0.7 + np.random.rand() * 1.5)
        img = TF.adjust_contrast(img, 0.5 + np.random.rand() * 1.5)
        img = TF.adjust_gamma(
            img, gamma=0.5 + np.random.rand(), gain=0.5 + np.random.rand()
        )
        img = TF.adjust_hue(img, -0.5 + np.random.rand())
        img = TF.adjust_saturation(img, np.random.rand() * 1.5)
        return img, kps

    @staticmethod
    def rotate(img: Image.Image, kps: Tensor) -> Tuple[Image.Image, Tensor]:
        rotation_angle_deg = np.random.rand() * 30 - 15
        rotation_angle_rad = np.deg2rad(rotation_angle_deg)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)],
            ]
        )
        rot_torch = torch.from_numpy(rotation_matrix.astype(np.float32))
        img = TF.rotate(img, np.rad2deg(rotation_angle_rad))

        center = torch.tensor([img.width, img.height]) / 2
        kps = kps - center
        kps = torch.matmul(kps, rot_torch)
        kps = kps + center
        return img, kps

    @staticmethod
    def perspective_augment(
        img: Image.Image, kps: Tensor
    ) -> Tuple[Image.Image, Tensor]:
        topleft = kps[0]
        topright = kps[1]
        bottomleft = kps[2]
        bottomright = kps[3]

        startpoints = [
            topleft.to(dtype=torch.int32).tolist(),
            topright.to(dtype=torch.int32).tolist(),
            bottomright.to(dtype=torch.int32).tolist(),
            bottomleft.to(dtype=torch.int32).tolist(),
        ]

        a = min(
            [
                torch.linalg.norm(topleft - topright) * 0.1,
                torch.linalg.norm(topleft - bottomleft) * 0.1,
            ]
        )
        new_topleft = topleft + (-a + np.random.rand() * 2 * a)
        new_topleft = torch.clip(
            new_topleft,
            torch.tensor([0, 0]),
            torch.tensor([img.width, img.height]),
        )
        new_topright = topright + (-a + np.random.rand() * 2 * a)
        new_topright = torch.clip(
            new_topright,
            torch.tensor([0, 0]),
            torch.tensor([img.width, img.height]),
        )
        new_bottomleft = bottomleft + (-a + np.random.rand() * 2 * a)
        new_bottomleft = torch.clip(
            new_bottomleft,
            torch.tensor([0, 0]),
            torch.tensor([img.width, img.height]),
        )
        new_bottomright = bottomright + (-a + np.random.rand() * 2 * a)
        new_bottomright = torch.clip(
            new_bottomright,
            torch.tensor([0, 0]),
            torch.tensor([img.width, img.height]),
        )

        endpoints = [
            new_topleft.to(dtype=torch.int32).tolist(),
            new_topright.to(dtype=torch.int32).tolist(),
            new_bottomright.to(dtype=torch.int32).tolist(),
            new_bottomleft.to(dtype=torch.int32).tolist(),
        ]
        img = transforms.functional.perspective(img, startpoints, endpoints)
        kps = torch.stack(
            [new_topleft, new_topright, new_bottomleft, new_bottomright]
        )

        return img, kps

    @staticmethod
    def crop_augment(
        img: Image.Image, kps: Tensor
    ) -> Tuple[Image.Image, Tensor]:
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
    def img_from_tensor(img_tensor: Tensor) -> Image.Image:
        img: np.ndarray = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase0PointsDataset.STD)
            + np.array(Phase0PointsDataset.MEAN)
        ) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def show(
        self, idx: int, out_folder: Path, repeat_idx=0, verbose: bool = False
    ):
        sample_t = self[idx]
        img_tensor = sample_t["img"]
        kps_tensor = sample_t["kps"]

        img = Phase0PointsDataset.img_from_tensor(img_tensor)
        kps = kps_tensor.reshape(-1, 2).numpy() * Phase0PointsDataset.IMG_SIZE
        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        save_img_with_kps(img, kps, filename, circle_radius=10, verbose=verbose)
