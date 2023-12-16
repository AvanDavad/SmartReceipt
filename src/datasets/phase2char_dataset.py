from src.draw_utils import draw_text_on_image, draw_vertical_line, put_stuffs_on_img

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def make_square_by_padding(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img

    if w > h:
        img_new = Image.new(img.mode, (w, w), color=(255, 255, 255))
        img_new.paste(img, (0, (w - h) // 2))
    else:
        img_new = Image.new(img.mode, (h, h), color=(255, 255, 255))
        img_new.paste(img, ((h - w) // 2, 0))

    return img_new


class Phase2CharDataset(Dataset):
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
            if "phase_2_3" not in self.image_reader[i]:
                continue

            s = self.image_reader[i]
            text = s["phase_2_3"]["text"]
            coords = s["phase_2_3"]["lines"]
            assert len(text) == len(coords) - 1

            num_chars = len(text)
            for j in range(num_chars):
                self.mapping.append((i, j, num_chars))

        self.idx_list = (
            np.random.permutation(len(self.mapping))
            if shuffle
            else np.arange(len(self.mapping))
        )

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        idx_mod = self.idx_list[idx]
        i, j, _ = self.mapping[idx_mod]
        s = self.image_reader[i]

        text = s["phase_2_3"]["text"]
        coords = s["phase_2_3"]["lines"]
        width = s["phase_1"]["img"].width

        x0 = coords[j]
        x1 = coords[j+1]
        ch = text[j]
        ch_ascii = ord(ch)

        num_prev_lines = int(np.floor(x0 / width))
        inside_line_idx = x0 - num_prev_lines * width

        img_orig: Image.Image = s["phase_1"]["img"]
        lines = s["phase_1"]["lines"]

        y_top = lines[num_prev_lines]
        y_bottom = lines[num_prev_lines+1]
        x_left = inside_line_idx
        x_right = x_left + (x1 - x0)

        y_top_img = y_top
        x_left_img = x_left

        if y_bottom - y_top > x_right - x_left:
            img_sidelen = y_bottom - y_top
        else:
            img_sidelen = x_right - x_left

        if self.augment:
            y_top_img = y_top_img + np.random.randint(-5, 5)
            x_left_img = x_left_img + np.random.randint(-5, 5)
            img_sidelen = img_sidelen + np.random.randint(-5, 5)

        img = img_orig.crop((x_left_img, y_top_img, x_left_img + img_sidelen, y_top_img + img_sidelen))

        img_tensor = Phase2CharDataset.TRANSFORMS(img)

        char_size = (x_right - x_left_img) / img_sidelen

        return img_tensor, ch_ascii, char_size

    @staticmethod
    def img_from_tensor(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (
            img * np.array(Phase2CharDataset.STD) + np.array(Phase2CharDataset.MEAN)
        ) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def show(self, idx, out_folder, repeat_idx=0):
        img_tensor, ch_ascii, char_size = self[idx]
        ch = f"<{chr(ch_ascii)}>"

        img = Phase2CharDataset.img_from_tensor(img_tensor)
        img = draw_text_on_image(img, text=ch, pos=(0,0))
        img = draw_vertical_line(img, int(img.width * char_size))

        filename = out_folder / f"sample_{idx}_{repeat_idx}.jpg"

        img.save(filename)
        print(f"Saved {filename}")
