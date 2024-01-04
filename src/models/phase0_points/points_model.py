from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from torch import Tensor

import wandb
from src.datasets.phase0points_dataset import Phase0PointsDataset
from src.draw_utils import make_square_by_cropping
from src.models.phase0_points.backbone import Phase0PointsBackbone
from src.visualization.font import get_font


class CNNModulePhase0Points(pl.LightningModule):
    def __init__(self, use_bn: bool = False):
        super().__init__()

        self.backbone = Phase0PointsBackbone(use_bn)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(32, 8)

        self.learning_rate = 2e-4
        self.weight_decay = 1e-5

        self.L2_weight = 1e5
        self.L1_weight = 1e3

    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        print(
            f"configure_optimizers. weight_decay: {self.weight_decay}, lr: {self.learning_rate}"
        )
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def configure_dropout_prob(self, dropout_prob: float):
        self.backbone.dropout = nn.Dropout(p=dropout_prob)

    def _loss(self, kps_gt, kps_pred):
        loss = (
            torch.mean((kps_gt - kps_pred) ** 2) * self.L2_weight
            + torch.mean(torch.abs(kps_gt - kps_pred)) * self.L1_weight
        )
        return loss

    def _pred_with_loss(self, batch):
        inp_img: Tensor = batch["img"]
        kps: Tensor = batch["kps"]
        kps_pred = self.forward(inp_img)

        loss = self._loss(kps, kps_pred)

        return kps_pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._pred_with_loss(batch)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"]
        wandb.log({"train_loss": train_loss.item()})

    def validation_step(self, batch, batch_idx):
        _, loss = self._pred_with_loss(batch)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        wandb.log({"val_loss": val_loss.item()})

    def inference(
        self,
        img: Image.Image,
        as_int: bool = False,
        to_tuple_list: bool = False,
    ) -> Union[np.ndarray, List[Tuple]]:
        assert img.width == img.height

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

    def inference_and_visualize(
        self,
        img: Image.Image,
        circle_radius: int = 5,
        circle_color: str = "blue",
        draw_index: bool = False,
        font_size: int = 25,
        font_color: str = "black",
        draw_lines: bool = True,
        line_width: int = 2,
        line_color: str = "blue",
    ) -> Image.Image:
        if not img.width == img.height:
            img = make_square_by_cropping(img)

        pred_kps = self.inference(img, as_int=True, to_tuple_list=True)

        draw = ImageDraw.Draw(img)

        for idx, kpt in enumerate(pred_kps):
            draw.ellipse(
                (
                    kpt[0] - circle_radius,
                    kpt[1] - circle_radius,
                    kpt[0] + circle_radius,
                    kpt[1] + circle_radius,
                ),
                fill=circle_color,
            )

            if draw_index:
                draw.text(
                    kpt, str(idx + 1), fill=font_color, font=get_font(font_size)
                )

        if draw_lines:
            for i0, i1 in [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
            ]:
                start_point = pred_kps[i0]
                end_point = pred_kps[i1]
                draw.line(
                    (start_point, end_point), fill=line_color, width=line_width
                )

        return img
