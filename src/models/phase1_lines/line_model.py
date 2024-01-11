from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

import wandb
from src.models.phase1_lines.backbone import Phase1LineBackbone


class CNNModulePhase1Line(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = Phase1LineBackbone()
        self.flatten = nn.Flatten()

        self.fc_line_y = nn.Linear(32, 1)

        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

        self.L2_weight = 1e5
        self.L1_weight = 1e3

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(batch["input"])
        x = self.flatten(x)
        pred_line_y = self.fc_line_y(x)
        return pred_line_y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def configure_dropout_prob(self, dropout_prob: float):
        self.backbone.dropout = nn.Dropout(p=dropout_prob)

    def _loss(self, gt_line_y, pred_line_y):
        loss = (
            torch.mean((gt_line_y - pred_line_y) ** 2) * self.L2_weight
            + torch.mean(torch.abs(gt_line_y - pred_line_y)) * self.L1_weight
        )
        return loss

    def _pred_with_loss(self, batch):
        pred_line_y = self.forward(batch)

        loss = self._loss(batch["line_y"], pred_line_y)

        return pred_line_y, loss

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

    # def inference_1(self, img: Image.Image, offset_y=0, mask_out_bottom=0):
    #     img_crop = img.crop((0, offset_y, img.width, offset_y + img.width))

    #     if mask_out_bottom > 0:
    #         img_crop = np.array(img_crop)
    #         img_crop[int(img.width / mask_out_bottom) :, ...] = 128
    #         img_crop = Image.fromarray(img_crop)

    #     input_img = Phase1LineDataset.TRANSFORMS(img_crop)
    #     input_img = input_img.unsqueeze(0)

    #     line_y, is_last_logit = self.forward(input_img)
    #     line_y = line_y.detach().numpy().flatten()[0] * img.width

    #     is_last_prob = sigmoid(is_last_logit).item()
    #     is_last = is_last_prob > 0.5

    #     return line_y, is_last, is_last_prob

    # def inference(
    #     self,
    #     img: Image.Image,
    #     out_folder: Optional[Path] = None,
    #     prefix="2_lines",
    #     max_num_lines=50,
    # ) -> List[Image.Image]:
    #     self.eval()
    #     line_img_list: List[Image.Image] = []
    #     offset_y = 0
    #     offset_y_list = []
    #     idx = 0

    #     while offset_y < img.height:
    #         line_y, _, is_last_prob = self.inference_1(img, offset_y=offset_y)

    #         print(f"line pred. line_y: {line_y}, is_last: {is_last_prob:.3f}")

    #         line_img = img.crop((0, offset_y, img.width, offset_y + line_y))
    #         line_img_list.append(line_img)

    #         offset_y += line_y
    #         offset_y_list.append(offset_y)

    #         idx += 1
    #         if idx >= max_num_lines:
    #             pass

    #     if out_folder is not None:
    #         lines = [(0, ly, img.width, ly) for ly in offset_y_list]
    #         img_with_line = put_stuffs_on_img(
    #             img,
    #             lines=lines,
    #             lines_colors="red",
    #             lines_width=2,
    #         )
    #         filename = out_folder / f"{prefix}_all.jpg"
    #         img_with_line.save(filename)
    #         print(f"Saved {filename}")

    #         for idx, line_img in enumerate(line_img_list):
    #             filename = out_folder / f"{prefix}_{str(idx).zfill(3)}.jpg"
    #             line_img.save(filename)
    #             print(f"Saved {filename}")

    #     return line_img_list
