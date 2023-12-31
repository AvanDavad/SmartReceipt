import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from typing import List, Tuple, Union
import torch.nn as nn

from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.visualization.font import get_font


class Phase0PointsBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(
            3, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_2 = nn.Conv2d(
            8, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_3 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_4 = nn.Conv2d(
            8, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_5 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_6 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_7 = nn.Conv2d(
            8, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_8 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_9 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_10 = nn.Conv2d(
            16, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_11 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_12 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_13 = nn.Conv2d(
            16, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_14 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_15 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_16 = nn.Conv2d(
            16, 32, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_17 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_18 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_19 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_20 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_21 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_22 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_23 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_24 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.conv_25 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_26 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_27 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.final_conv = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="valid", bias=True
        )

    def forward(self, x):
        x = x0 = self.conv_1(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = self.conv_2(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = self.conv_3(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = x0 + x  # 768x768x8

        x = x0 = self.avg_pool(x)  # 384x384x8
        x = self.conv_4(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = self.conv_5(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = self.conv_6(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = x0 + x  # 384x384x8

        x = self.avg_pool(x)  # 192x192x8
        x = x0 = self.conv_7(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = self.conv_8(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = self.conv_9(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = x0 + x  # 192x192x16

        x = x0 = self.avg_pool(x)  # 96x96x16
        x = self.conv_10(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = self.conv_11(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = self.conv_12(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = x0 + x  # 96x96x16

        x = x0 = self.avg_pool(x)  # 48x48x16
        x = self.conv_13(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = self.conv_14(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = self.conv_15(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = x0 + x  # 48x48x16

        x = self.avg_pool(x)  # 24x24x16
        x = x0 = self.conv_16(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = self.conv_17(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = self.conv_18(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = x0 + x  # 24x24x32

        x = x0 = self.avg_pool(x)  # 12x12x32
        x = self.conv_19(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = self.conv_20(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = self.conv_21(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = x0 + x  # 12x12x32

        x = x0 = self.avg_pool(x)  # 6x6x32
        x = self.conv_22(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = self.conv_23(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = self.conv_24(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = x0 + x  # 6x6x32

        x = x0 = self.avg_pool(x)  # 3x3x32
        x = self.conv_25(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = self.conv_26(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = self.conv_27(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = x0 + x  # 3x3x32

        x = self.final_conv(x)  # 1x1x32

        return x


class CNNModulePhase0Points(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = Phase0PointsBackbone()
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(1 * 1 * 32, 8)

        self.learning_rate = 2e-4
        self.weight_decay = 1e-5

        self.L2_weight = 1e5
        self.L1_weight = 1e3

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _loss(self, kps_gt, kps_pred):
        loss = torch.mean((kps_gt - kps_pred) ** 2) * self.L2_weight + torch.mean(torch.abs(kps_gt - kps_pred)) * self.L1_weight
        return loss

    def training_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self._loss(kps, kps_pred)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self._loss(kps, kps_pred)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def inference(self, img: Image.Image, as_int: bool = False, to_tuple_list: bool = False) -> Union[np.ndarray, List[Tuple]]:
        img_tensor: Tensor = Phase0PointsDataset.TRANSFORMS(img)
        img_tensor = img_tensor.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pred_kps: Tensor = self(img_tensor)

        pred_kps_np: np.ndarray = pred_kps.detach().cpu().numpy()[0]
        pred_kps_np = pred_kps_np.reshape(-1, 2)
        pred_kps_np *= np.array([img.width, img.height])

        if as_int:
            pred_kps_np = pred_kps_np.astype(int)

        if to_tuple_list:
            pred_kps_tup = [(x[0], x[1]) for x in pred_kps_np]
            return pred_kps_tup

        return pred_kps_np
