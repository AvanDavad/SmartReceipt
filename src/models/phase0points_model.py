import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights


from pathlib import Path
from src.datasets.phase0points_dataset import Phase0PointsDataset

from src.visualization.font import get_font

class Phase0PointsBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_10 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_13 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_14 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_15 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        x1 = self.conv_1(x) # 112x112x16
        x2 = self.relu(x1) # 112x112x16
        x3 = self.conv_2(x2) # 112x112x16
        x4 = self.relu(x3) # 112x112x16
        x5 = self.conv_3(x4) # 112x112x16
        x6 = self.relu(x5) + x1

        x7 = self.conv_4(x6) # 56x56x32
        x8 = self.relu(x7) # 56x56x32
        x9 = self.conv_5(x8) # 56x56x32
        x10 = self.relu(x9) # 56x56x32
        x11 = self.conv_6(x10) # 56x56x32
        x12 = self.relu(x11) + x7

        x13 = self.conv_7(x12) # 28x28x64
        x14 = self.relu(x13) # 28x28x64
        x15 = self.conv_8(x14) # 28x28x64
        x16 = self.relu(x15) # 28x28x64
        x17 = self.conv_9(x16) # 28x28x64
        x18 = self.relu(x17) + x13

        x19 = self.conv_10(x18) # 14x14x64
        x20 = self.relu(x19) # 14x14x64
        x21 = self.conv_11(x20) # 14x14x64
        x22 = self.relu(x21) # 14x14x64
        x23 = self.conv_12(x22) # 14x14x64
        x24 = self.relu(x23) + x19

        x25 = self.conv_13(x24) # 7x7x96
        x26 = self.relu(x25) # 7x7x96
        x27 = self.conv_14(x26) # 7x7x96
        x28 = self.relu(x27) # 7x7x96
        x29 = self.conv_15(x28) # 7x7x96
        x30 = self.relu(x29) + x25

        return x30

class CNNModulePhase0Points(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = Phase0PointsBackbone()
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(7*7*96, 8)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = torch.mean((kps - kps_pred) ** 2)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = torch.mean((kps - kps_pred) ** 2)

        self.log("val_loss", loss, prog_bar=True)

    def inference(self, img_path, out_folder=Path(""), prefix="inference"):
        img = Image.open(img_path)
        img_tensor = Phase0PointsDataset.TRANSFORMS(img)
        img_tensor = img_tensor.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pred_kps = self(img_tensor)
        pred_kps = pred_kps.detach().cpu().numpy()[0]

        draw = ImageDraw.Draw(img)

        keypoints = []
        for i in range(pred_kps.shape[0] // 2):
            kpt = (
                int(img.width * pred_kps[2 * i]),
                int(img.height * pred_kps[2 * i + 1]),
            )
            keypoints.append(kpt)
            circle_radius = 20
            circle_color = "blue"
            draw.ellipse(
                (
                    kpt[0] - circle_radius,
                    kpt[1] - circle_radius,
                    kpt[0] + circle_radius,
                    kpt[1] + circle_radius,
                ),
                fill=circle_color,
            )

            # Draw text
            text = f"kpt_{i+1}"
            font_size = 100
            text_color = "black"
            font = get_font(font_size)
            text_position = kpt
            draw.text(text_position, text, fill=text_color, font=font)

        for i0, i1, col in [(0, 1, "red"), (0, 2, "red"), (1, 3, "red"), (2, 3, "yellow")]:
            start_point = keypoints[i0]
            end_point = keypoints[i1]
            line_width = 5
            draw.line((start_point, end_point), fill=col, width=line_width)

        # Save the image to a file
        prefix = f"{prefix}_" if prefix else ""
        filename = out_folder / f"{prefix}{img_path.stem}.jpg"
        img.save(filename)
        print(f"saved {filename}")
        return np.array(keypoints).astype(np.float64)
