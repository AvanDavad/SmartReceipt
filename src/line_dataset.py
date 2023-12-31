import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.camera_calib import get_camera_calib
from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.draw_utils import save_img_with_kps
from src.warp_perspective import warp_perspective_with_nonlin_least_squares


class LineDataset(Dataset):
    def __init__(self, image_reader, augment=False, shuffle=False):
        self.image_reader = image_reader
        self.augment = augment

        self.mapping = []
        for i in range(len(self.image_reader)):
            s = self.image_reader[i]
            num_lines = len(s["phase_1"]["lines"]) - 1
            for j in range(num_lines):
                self.mapping.append((i, j, num_lines))

        self.idx_list = (
            np.random.permutation(len(self.mapping))
            if shuffle
            else np.arange(len(self.mapping))
        )

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        idx_mod = self.idx_list[idx]
        i, j, num_lines = self.mapping[idx_mod]
        s = self.image_reader[i]

        img = s["phase_1"]["img"]
        line_kps = np.array(s["lines"])

        is_last = j == num_lines - 1
        is_first = j == 0

        base_kps = np.array(s["keypoints"])
        if self.augment:
            base_kps += np.random.randn(*base_kps.shape) * 5.0

        camera_matrix, dist_coeffs = get_camera_calib()
        img, base_kps, M = warp_perspective_with_nonlin_least_squares(
            img,
            base_kps,
            camera_matrix,
            dist_coeffs,
            scale_factor=10.0,
            verbose=False,
        )
        line_kps = cv2.perspectiveTransform(
            line_kps.reshape(-1, 1, 2), M
        ).reshape(-1, 2)
        curr_line_kps = line_kps[2 * j : 2 * j + 4]

        crop_x0_max = max(
            1, min([int(base_kps[0, 0]), int(curr_line_kps[:, 0].min())])
        )
        crop_x0 = np.random.randint(0, crop_x0_max) if self.augment else 0
        crop_x1_min = min(
            img.width - 1,
            max([int(base_kps[3, 0]), int(curr_line_kps[:, 0].max())]),
        )
        crop_x1 = (
            np.random.randint(crop_x1_min, img.width)
            if self.augment
            else img.width
        )
        if is_first:
            y_min = int(base_kps[:4, 1].min())
            y_max = max(int(line_kps[:2, 1].min()), y_min + 1)
            crop_y0 = np.random.randint(y_min, y_max) if self.augment else y_min
        else:
            prev_line_kps = line_kps[2 * j - 2 : 2 * j + 2]
            y_min = int(prev_line_kps[:, 1].mean())
            y_max = max(int(curr_line_kps[:, 1].min()), y_min + 1)
            crop_y0 = np.random.randint(y_min, y_max) if self.augment else y_min
        crop_y1 = crop_y0 + crop_x1 - crop_x0

        img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

        # draw the bottom part of the image black
        img = np.array(img)
        ratio = np.random.randint(3, 10) if self.augment else 6
        img[int(img.shape[0] / ratio) :, :, :] = 128
        img = Image.fromarray(img)
        assert img.width == img.height, img.size

        curr_line_kps = curr_line_kps - np.array([crop_x0, crop_y0])

        curr_line_kps = curr_line_kps / np.array([img.width, img.height])
        curr_line_kps = curr_line_kps.flatten()
        curr_line_kps = torch.from_numpy(curr_line_kps)

        img_tensor = Phase0PointsDataset.TRANSFORMS(img)

        is_last = torch.tensor([is_last]).float()
        return img_tensor, curr_line_kps, is_last

    def show(self, idx, out_folder, repeat_idx=0):
        img_tensor, curr_line_kps, is_last = self[idx]

        img = Phase0PointsDataset.img_from_tensor(img_tensor)

        curr_line_kps = (
            curr_line_kps.reshape(-1, 2).numpy() * Phase0PointsDataset.IMG_SIZE
        )

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"
        color = "red" if (is_last.item() == 1.0) else "yellow"
        save_img_with_kps(
            img, curr_line_kps, filename, circle_color=color, circle_radius=1
        )
