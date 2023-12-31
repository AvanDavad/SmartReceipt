import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.draw_utils import put_stuffs_on_img


class Phase1LineDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.2, 0.2, 0.2]
    IMG_SIZE = 384
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

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

        is_last = j == num_lines - 1
        is_first = j == 0
        lines = s["phase_1"]["lines"]

        y_offset_min = (
            (lines[j] - 20 if is_first else (lines[j - 1] + lines[j] * 2) // 3)
            if self.augment
            else lines[j]
        )
        y_offset_max = (
            (lines[j] + 20 if is_last else (lines[j] * 2 + lines[j + 1]) // 3)
            if self.augment
            else lines[j] + 1
        )
        y_offset = np.random.randint(y_offset_min, y_offset_max)

        img_cropped = img.crop((0, y_offset, img.width, y_offset + img.width))

        if self.augment:
            # draw the bottom part of the image black
            cut_y = np.random.randint(
                Phase1LineDataset.IMG_SIZE // 5, Phase1LineDataset.IMG_SIZE
            )
            img_cropped_np = np.array(img_cropped)
            img_cropped_np[cut_y:, ...] = 0
            img_cropped = Image.fromarray(img_cropped_np)

        img_tensor = Phase1LineDataset.TRANSFORMS(img_cropped)
        line_y = torch.tensor([(lines[j + 1] - y_offset) / img.width]).float()
        is_last = torch.tensor([is_last]).float()

        return img_tensor, line_y, is_last

    @staticmethod
    def img_from_tensor(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase1LineDataset.STD)
            + np.array(Phase1LineDataset.MEAN)
        ) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def show(self, idx, out_folder, repeat_idx=0):
        img_tensor, line_y, is_last = self[idx]

        img = Phase1LineDataset.img_from_tensor(img_tensor)

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"
        color = "red" if (is_last.item() == 1.0) else "yellow"
        line = (
            0,
            int(line_y.item() * img.width),
            img.width,
            int(line_y.item() * img.width),
        )

        img = put_stuffs_on_img(
            img, lines=[line], lines_colors=color, lines_width=2
        )
        img.save(filename)
        print(f"Saved {filename}")
