from pathlib import Path
from typing import Dict, Optional, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.datasets.phase2char_dataset import Phase2CharDataset
from src.datasets.phase2char_dataset import ALL_CHARS
from torch import Tensor
from PIL import Image

from src.draw_utils import draw_for_char_recognition

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2, bias=True)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_final = nn.Conv2d(128, 128, kernel_size=2, padding="valid", bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert Phase2CharDataset.IMG_SIZE == 128
        assert x.shape == (bs, 3, 128, 128)

        x = self.conv_1(x) # 64x64 channels: 8

        x = self.avg_pool(x) # 32x32 channels: 8
        x = x0 = self.conv_2(x) # 32x32 channels: 16
        x = self.relu(x) # 32x32 channels: 16
        x = x0 + x # 32x32 channels: 16

        x = self.avg_pool(x) # 16x16 channels: 16
        x = x0 = self.conv_3(x) # 16x16 channels: 32
        x = self.relu(x) # 16x16 channels: 32
        x = x0 + x # 16x16 channels: 32

        x = self.avg_pool(x) # 8x8 channels: 32
        x = x0 = self.conv_4(x) # 8x8 channels: 64
        x = self.relu(x) # 8x8 channels: 64
        x = x0 + x # 8x8 channels: 64

        x = self.avg_pool(x) # 4x4 channels: 64
        x = x0 = self.conv_5(x) # 4x4 channels: 128
        x = self.relu(x) # 4x4 channels: 128
        x = x0 + x # 4x4 channels: 128

        x = self.avg_pool(x) # 2x2 channels: 128
        x = self.conv_final(x) # 1x1 channels: 128

        return x

class ImageFeatureReduce(nn.Module):
    def __init__(self, full_h: int = 128, tiny_h: int = 16):
        super().__init__()
        self.full_h = full_h

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(full_h, full_h//2)
        self.fc_2 = nn.Linear(full_h//2, full_h//4)
        self.fc_3 = nn.Linear(full_h//4, tiny_h)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert x.shape == (bs, self.full_h)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)

        return x

class FCNForChars(nn.Module):
    def __init__(self, w: int = 5, tiny_h: int = 16, full_h: int = 128, num_classes: int = len(ALL_CHARS) + 1):
        super().__init__()

        self.in_dim = w * tiny_h + full_h + w * tiny_h
        self.num_classes = num_classes

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(self.in_dim, (self.in_dim + num_classes)//2)
        self.fc_2 = nn.Linear((self.in_dim + num_classes)//2, (self.in_dim + num_classes)//2)
        self.fc_3 = nn.Linear((self.in_dim + num_classes)//2, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert x.shape == (bs, self.in_dim)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)

        return x

class FCNForCharWidth(nn.Module):
    def __init__(self, full_h: int = 128):
        super().__init__()

        self.in_dim = full_h

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(self.in_dim, self.in_dim//2)
        self.fc_2 = nn.Linear(self.in_dim//2, self.in_dim//4)
        self.fc_3 = nn.Linear(self.in_dim//4, 1)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert x.shape == (bs, self.in_dim)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)

        return x

class CNNModulePhase2Chars(pl.LightningModule):
    def __init__(self, w: int = 5, tiny_h: int = 16, full_h: int = 128, num_classes: int = len(ALL_CHARS) + 1):
        super().__init__()

        self.cnn_backbone = CNNBackbone()
        self.flatten = nn.Flatten()

        self.image_feature_reduce = ImageFeatureReduce(full_h=full_h, tiny_h=tiny_h)
        self.last_fcn = FCNForChars(w=w, tiny_h=tiny_h, full_h=full_h, num_classes=num_classes)

        self.fcn_for_char_width = FCNForCharWidth()

        self.ce = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["img"]
        bs, w2p1, ch, height, width = images.shape
        assert height == width
        assert ch == 3
        w = (w2p1 - 1) // 2
        images = images.view((bs * w2p1, ch, height, width))
        image_features: Tensor = self.cnn_backbone(images)
        image_features = image_features.view((bs * w2p1, 128))

        tiny_image_features: Tensor = self.image_feature_reduce(image_features)
        tiny_image_features = tiny_image_features.view((bs, w2p1, 16))
        tiny_pre_features = tiny_image_features[:, :w, :].view((bs, -1))
        tiny_post_features = tiny_image_features[:, w + 1:, :].view((bs, -1))

        image_features = image_features.view((bs, w2p1, 128))
        actual_image_feature = image_features[:, w]

        concat_image_features = torch.concat([tiny_pre_features, actual_image_feature, tiny_post_features], dim=1)

        chars_logits = self.last_fcn(concat_image_features)
        char_width = self.fcn_for_char_width(actual_image_feature)

        return chars_logits, char_width

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=1e-5)
        return optimizer

    def _loss(self, batch: Dict[str, Tensor], chars_logits: Tensor, char_width: Tensor, name: str):

        char_loss = self.ce(chars_logits, batch["label_idx"].flatten())

        char_width_gt = batch["char_width"].flatten()
        is_double_space_gt = batch["is_double_space"].flatten()
        width_factor = torch.where(is_double_space_gt, 0.01, 1.0)
        loss_width = torch.abs(char_width_gt - char_width.flatten()) * width_factor
        loss_width = torch.mean(loss_width)

        loss = char_loss + loss_width

        self.log(name, loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        chars_logits, char_width = self.forward(batch)

        return self._loss(batch, chars_logits, char_width, "train_loss")

    def validation_step(self, batch, batch_idx):
        chars_logits, char_width = self.forward(batch)

        return self._loss(batch, chars_logits, char_width, "val_loss")

    def inference(self, line_image: Image.Image, out_folder: Optional[Path] = None, prefix: str="", w: int = 5, verbose: bool = False) -> str:
        feature_list = []
        tiny_feature_list = []
        x_offset_list = []
        x_offset = 0.0
        while x_offset < line_image.width:
            img = line_image.crop((x_offset, 0, x_offset + line_image.height, line_image.height))
            input_img: Tensor = Phase2CharDataset.TRANSFORMS(img)
            image_feature: Tensor = self.cnn_backbone(input_img.unsqueeze(0))
            feature_list.append(image_feature.flatten())
            char_width_relative = self.fcn_for_char_width(image_feature[:, :, 0, 0]).item()
            char_width = max(int(char_width_relative * line_image.height), 1)
            x_offset += char_width
            x_offset_list.append(x_offset)

            tiny_feature: Tensor = self.image_feature_reduce(image_feature[:, :, 0, 0])
            tiny_feature_list.append(tiny_feature.flatten())

        black_image = Image.new("RGB", (line_image.height, line_image.height), color=(0, 0, 0))
        black_feature = self.cnn_backbone(Phase2CharDataset.TRANSFORMS(black_image).unsqueeze(0))
        tiny_black_feature = self.image_feature_reduce(black_feature[:, :, 0, 0]).flatten()

        char_list = []
        confidence_list = []
        for idx in range(len(feature_list)):
            tiny_pre_features = []
            num_black_features = max(0, w - idx)
            num_not_black_features = w - num_black_features
            for _ in range(num_black_features):
                tiny_pre_features.append(tiny_black_feature)
            for jj in range(num_not_black_features):
                tiny_pre_features.append(tiny_feature_list[idx - num_not_black_features + jj])

            tiny_post_features = []
            num_black_features = max(0, w - (len(feature_list) - idx - 1))
            num_not_black_features = w - num_black_features
            for jj in range(num_not_black_features):
                tiny_post_features.append(tiny_feature_list[idx + 1 + jj])
            for _ in range(num_black_features):
                tiny_post_features.append(tiny_black_feature)

            all_features = torch.cat(tiny_pre_features + [feature_list[idx]] + tiny_post_features, dim=0).unsqueeze(0)
            chars_logits = self.last_fcn(all_features)
            char_idx = torch.argmax(chars_logits).item()
            char = ALL_CHARS[char_idx]
            char_list.append(char)
            probabilities = torch.softmax(chars_logits, dim=1).flatten()
            confidence = probabilities[char_idx].item()
            confidence_list.append(confidence)

        line_text = "".join(char_list)

        if out_folder is not None:
            filename = out_folder / f"{prefix}_line.jpg"

            img_line = draw_for_char_recognition(
                img=line_image,
                offset_list=x_offset_list,
                char_list=char_list,
                confidence_list=confidence_list,
            )
            img_line.save(filename)
            if verbose:
                print(f"Saved {filename}")

        return line_text
