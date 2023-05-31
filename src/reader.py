import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from src.camera_calib import get_camera_calib
from src.draw_utils import save_img_with_kps

from src.warp_perspective import warp_perspective

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
        filename = f"sample_{idx}.jpg"
        plt.savefig(filename)
        print(f"saved {filename}")

class ImageDataset(Dataset):
    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
            img, kps = ImageDataset.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = ImageDataset.transforms(img)

        return img_tensor, kps

    @staticmethod
    def crop_augment(img, kps):
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


class Top2PointsDataset(Dataset):
    def __init__(self, reader, augment=False, zoom_in_factor=5.0):
        self.reader = reader
        self.augment = augment
        self.zoom_in_factor = zoom_in_factor

    def __len__(self):
        return 2*len(self.reader)

    def __getitem__(self, idx):
        s = self.reader[idx // 2]

        img = s["img"]
        offset = 2 * (idx % 2)
        kps = torch.tensor(s["keypoints"])[offset:offset+2]

        img, kps = Top2PointsDataset.zoom_in(img, kps, self.zoom_in_factor)

        # augment
        if self.augment:
            img, kps = ImageDataset.crop_augment(img, kps)

        kps = kps / torch.tensor([img.width, img.height])
        kps = kps.flatten()

        img_tensor = ImageDataset.transforms(img)

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

class LineDataset(Dataset):
    def __init__(self, reader, augment=False, shuffle=False, debug=False):
        self.reader = reader
        self.augment = augment
        self.debug = debug

        self.mapping = []
        for i in range(len(self.reader)):
            s = self.reader[i]
            if "lines" in s:
                num_lines = len(s["lines"]) // 2 - 1
                for j in range(num_lines):
                    self.mapping.append((i, j, num_lines))
        
        if shuffle:
            self.idx_list = np.random.permutation(len(self.mapping))
        else:
            self.idx_list = np.arange(len(self.mapping))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        idx_mod = self.idx_list[idx]
        i, j, num_lines = self.mapping[idx_mod]
        s = self.reader[i]

        img = s["img"]
        line_kps = np.array(s["lines"])

        is_last = (j == num_lines - 1)
        is_first = (j == 0)

        base_kps = np.array(s["keypoints"])
        if self.augment:
            base_kps += np.random.randn(*base_kps.shape) * 5.0

        camera_matrix, dist_coeffs = get_camera_calib()
        img, base_kps, M = warp_perspective(img, base_kps, camera_matrix, dist_coeffs, scale_factor=10.0, verbose=False)
        line_kps = cv2.perspectiveTransform(line_kps.reshape(-1, 1, 2), M).reshape(-1, 2)
        curr_line_kps = line_kps[2*j : 2*j+4]

        crop_x0_max = max(1, min([int(base_kps[0,0]), int(curr_line_kps[:,0].min())]))
        crop_x0 = np.random.randint(0, crop_x0_max) if self.augment else 0
        crop_x1_min = min(img.width-1, max([int(base_kps[3,0]), int(curr_line_kps[:,0].max())]))
        crop_x1 = np.random.randint(crop_x1_min, img.width) if self.augment else img.width
        if is_first:
            y_min = int(base_kps[:4, 1].min())
            y_max = max(int(line_kps[:2,1].min()), y_min+1)
            crop_y0 = np.random.randint(y_min, y_max) if self.augment else y_min
        else:
            prev_line_kps = line_kps[2*j-2 : 2*j+2]
            y_min = int(prev_line_kps[:,1].mean())
            y_max = max(int(curr_line_kps[:,1].min()), y_min+1)
            crop_y0 = np.random.randint(y_min, y_max) if self.augment else y_min
        crop_y1 = crop_y0 + crop_x1 - crop_x0

        img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

        # draw the bottom part of the image black
        img = np.array(img)
        ratio = np.random.randint(3, 10) if self.augment else 6
        img[int(img.shape[0] / ratio):, :, :] = 128
        img = Image.fromarray(img)
        assert img.width == img.height, img.size

        curr_line_kps = curr_line_kps - np.array([crop_x0, crop_y0])

        if self.debug:
            i = 0
            filename = Path("debug") / f"img_{i}.jpg"
            while filename.is_file():
                i += 1
                filename = Path("debug") / f"img_{i}.jpg"
            color = "lightgreen" if is_first else ("red" if is_last else "yellow")
            save_img_with_kps(img, curr_line_kps, filename, circle_color=color)

        curr_line_kps = curr_line_kps / np.array([img.width, img.height])
        curr_line_kps = curr_line_kps.flatten()
        curr_line_kps = torch.from_numpy(curr_line_kps)

        img_tensor = ImageDataset.transforms(img)

        is_last = torch.tensor([is_last]).float()
        return img_tensor, curr_line_kps, is_last

# dataset = LineDataset(ImageReader("/home/avandavad/projects/receipt_extractor/data/train"), augment=True, shuffle=True, debug=True)
# for i in range(len(dataset)):
#     dataset[i]