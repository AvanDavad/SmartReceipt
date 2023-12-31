from typing import Dict, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.datasets.phase2char_border_dataset import Phase2CharBorderDataset
from torch import Tensor
from PIL import Image
from typing import List


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_final = nn.Conv2d(32, 32, kernel_size=4, padding="valid", bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert Phase2CharBorderDataset.IMG_SIZE == 64
        assert x.shape == (bs, 3, 64, 64)

        x = self.conv_1(x) # 64x64 channels: 4

        x = self.avg_pool(x) # 32x32 channels: 4
        x = self.conv_2(x) # 32x32 channels: 8
        x = self.relu(x) # 32x32 channels: 8

        x = self.avg_pool(x) # 16x16 channels: 8
        x = self.conv_3(x) # 16x16 channels: 16
        x = self.relu(x) # 16x16 channels: 16

        x = self.avg_pool(x) # 8x8 channels: 16
        x = self.conv_4(x) # 8x8 channels: 32
        x = self.relu(x) # 8x8 channels: 32

        x = self.avg_pool(x) # 4x4 channels: 32
        x = self.conv_final(x) # 1x1 channels: 32

        return x

class CNNModulePhase2CharsBorder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.cnn_backbone = CNNBackbone()
        self.flatten = nn.Flatten()

        self.fcn = nn.Linear(32, 1)

        self.lr = 1e-3
        self.weight_decay = 1e-5

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["img"]
        image_features = self.cnn_backbone(images)
        image_features = self.flatten(image_features)
        char_end_x = self.fcn(image_features)

        return char_end_x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _loss(self, batch: Dict[str, Tensor], char_end_x: Tensor, name: str):

        gt_char_end_x = batch["char_end_x"]

        loss = (char_end_x - gt_char_end_x) ** 2 + 10.0 * torch.abs(char_end_x - gt_char_end_x)
        weight = torch.where(batch["is_double_space"], 1e-6, 1.0)
        weighted_loss = loss * weight

        mean_loss = torch.mean(weighted_loss)

        self.log(name, mean_loss, prog_bar=True)

        return mean_loss

    def training_step(self, batch, batch_idx):
        char_end_x = self.forward(batch)

        return self._loss(batch, char_end_x, "train_loss")

    def validation_step(self, batch, batch_idx):
        char_end_x = self.forward(batch)

        return self._loss(batch, char_end_x, "val_loss")

    def inference(self, line_image: Image.Image) -> List[float]:
        height = line_image.height

        x_coords = [0.0]

        while x_coords[-1] < line_image.width:
            img_crop = line_image.crop((x_coords[-1], 0, x_coords[-1] + height, height))

            input_tensor = Phase2CharBorderDataset.TRANSFORMS(img_crop).unsqueeze(0)
            char_end_x = self.forward({"img": input_tensor}).item()

            x_coords.append(x_coords[-1] + char_end_x * height)

        return x_coords