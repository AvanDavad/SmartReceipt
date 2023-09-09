from pathlib import Path
from typing import Optional, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import sigmoid
from PIL import Image
from src.draw_utils import put_stuffs_on_img
from src.datasets.phase1line_dataset import Phase1LineDataset



class Phase1LineBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_5 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_9 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_10 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_11 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_13 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_14 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_15 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_16 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=True)

        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="valid", bias=True)

    def forward(self, x):

        x = x0 = self.conv_1(x) # 384x384x8
        x = self.relu(x) # 384x384x8
        x = self.conv_2(x) # 384x384x8
        x = self.relu(x) # 384x384x8
        x = x0 + x # 384x384x8

        x = x0 = self.avg_pool(x) # 192x192x8
        x = self.conv_3(x) # 192x192x8
        x = self.relu(x) # 192x192x8
        x = self.conv_4(x) # 192x192x8
        x = self.relu(x) # 192x192x8
        x = x0 + x

        x = self.avg_pool(x) # 96x96x8
        x = x0 = self.conv_5(x) # 96x96x16
        x = self.relu(x) # 96x96x16
        x = self.conv_6(x) # 96x96x16
        x = self.relu(x) # 96x96x16
        x = x0 + x

        x = x0 = self.avg_pool(x) # 48x48x16
        x = self.conv_7(x) # 48x48x16
        x = self.relu(x) # 48x48x16
        x = self.conv_8(x) # 48x48x16
        x = self.relu(x) # 48x48x16
        x = x0 + x

        x = x0 = self.avg_pool(x) # 24x24x16
        x = self.conv_9(x) # 24x24x16
        x = self.relu(x) # 24x24x16
        x = self.conv_10(x) # 24x24x16
        x = self.relu(x) # 24x24x16
        x = x0 + x

        x = self.avg_pool(x) # 12x12x16
        x = x0 = self.conv_11(x) # 12x12x32
        x = self.relu(x) # 12x12x32
        x = self.conv_12(x) # 12x12x32
        x = self.relu(x) # 12x12x32
        x = x0 + x

        x = x0 = self.avg_pool(x) # 6x6x32
        x = self.conv_13(x) # 6x6x32
        x = self.relu(x) # 6x6x32
        x = self.conv_14(x) # 6x6x32
        x = self.relu(x) # 6x6x32
        x = x0 + x

        x = x0 = self.avg_pool(x) # 3x3x32
        x = self.conv_15(x) # 3x3x32
        x = self.relu(x) # 3x3x32
        x = self.conv_16(x) # 3x3x32
        x = self.relu(x) # 3x3x32
        x = x0 + x

        x = self.final_conv(x) # 1x1x32

        return x

class CNNModulePhase1Line(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = Phase1LineBackbone()
        self.flatten = nn.Flatten()

        self.fc_line_y = nn.Linear(1*1*32, 1)
        self.fc_is_last = nn.Linear(1*1*32, 1)

        self.bce_loss_module = nn.BCEWithLogitsLoss()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = self.flatten(x)
        line_y_pred = self.fc_line_y(x)
        is_last_pred = self.fc_is_last(x)
        return line_y_pred, is_last_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=1e-5)
        return optimizer

    def _loss(self, line_y_pred, is_last_pred, line_y_gt, is_last_gt, name, factor=1.0):
        kps_loss = torch.mean((line_y_gt - line_y_pred) ** 2)
        cls_loss = self.bce_loss_module(is_last_pred, is_last_gt)
        loss = kps_loss + cls_loss

        self.log(name, factor*loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        inp_img, line_y_gt, is_last_gt = batch
        line_y_pred, is_last_pred = self(inp_img)

        return self._loss(line_y_pred, is_last_pred, line_y_gt, is_last_gt, "train_loss")

    def validation_step(self, batch, batch_idx):
        inp_img, line_y_gt, is_last_gt = batch
        line_y_pred, is_last_pred = self(inp_img)

        return self._loss(line_y_pred, is_last_pred, line_y_gt, is_last_gt, "val_loss", factor=1e5)

    def inference_1(self, img: Image.Image, offset_y=0, mask_out_bottom = 0):
        img_crop = img.crop((0, offset_y, img.width, offset_y + img.width))

        if mask_out_bottom > 0:
            img_crop = np.array(img_crop)
            img_crop[int(img.width / mask_out_bottom) :, ...] = 128
            img_crop = Image.fromarray(img_crop)

        input_img = Phase1LineDataset.TRANSFORMS(img_crop)
        input_img = input_img.unsqueeze(0)

        line_y, is_last_logit = self.forward(input_img)
        line_y = line_y.detach().numpy().flatten()[0] * img.width

        is_last_prob = sigmoid(is_last_logit).item()
        is_last = is_last_prob > 0.5

        return line_y, is_last, is_last_prob

    def inference(self, img: Image.Image, out_folder:Optional[Path]=None, prefix="2_lines", max_num_lines=50):
        self.eval()
        is_last = False
        line_img_list = []
        offset_y = 0
        offset_y_list = []
        idx = 0

        while not is_last:
            if offset_y >= img.height:
                break

            line_y, is_last, is_last_prob = self.inference_1(img, offset_y=offset_y)

            print(f"line pred. line_y: {line_y}, is_last: {is_last_prob:.3f}")

            line_img = img.crop((0, offset_y, img.width, offset_y + line_y))
            line_img_list.append(line_img)

            offset_y += line_y
            offset_y_list.append(offset_y)

            idx += 1
            if idx >= max_num_lines:
                is_last = True

        if out_folder is not None:

            lines = [(0, ly, img.width, ly) for ly in offset_y_list]
            img_with_line = put_stuffs_on_img(
                img,
                lines=lines,
                lines_colors="red",
                lines_width=2,
            )
            filename = out_folder / f"{prefix}_all.jpg"
            img_with_line.save(filename)
            print(f"Saved {filename}")

            for idx, line_img in enumerate(line_img_list):
                filename = out_folder / f"{prefix}_{str(idx).zfill(3)}.jpg"
                line_img.save(filename)
                print(f"Saved {filename}")
