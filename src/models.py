from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision.models import ResNet18_Weights


class CNNModule6Points(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet model (e.g., ResNet-18)
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the last fully connected layer to match the desired output size
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 12)
        self.loss_module = nn.L1Loss()

        for param in self.backbone.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self.loss_module(kps, kps_pred)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self.loss_module(kps, kps_pred)

        self.log("val_loss", loss, prog_bar=True)

    def inference(self, img_path, preproc, out_folder=Path(""), prefix="inference"):
        img = Image.open(img_path)
        img_tensor = preproc(img)
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
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                font_size,
            )
            text_position = kpt
            draw.text(text_position, text, fill=text_color, font=font)

        for i0, i1, col in [(0, 1, "red"), (2, 3, "red"), (4, 5, "yellow")]:
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


class CNNModule2Points(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 4)
        self.loss_module = nn.L1Loss()

        for param in self.backbone.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self.loss_module(kps, kps_pred)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps = batch
        kps_pred = self(inp_img)

        loss = self.loss_module(kps, kps_pred)

        self.log("val_loss", loss, prog_bar=True)




class CNNModuleLineDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 9)

        self.kps_loss_module = nn.L1Loss()
        self.bce_loss_module = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.backbone(x)
        kps = out[:, :8]
        is_last = out[:, 8:9]
        return kps, is_last

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_img, kps, is_last = batch

        kps_pred, is_last_pred = self(inp_img)

        kps_loss = self.kps_loss_module(kps, kps_pred)
        cls_loss = self.bce_loss_module(is_last_pred, is_last)
        loss = kps_loss + cls_loss

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps, is_last = batch

        kps_pred, is_last_pred = self(inp_img)

        kps_loss = self.kps_loss_module(kps, kps_pred)
        cls_loss = self.bce_loss_module(is_last_pred, is_last)
        loss = kps_loss + cls_loss

        self.log("val_loss", loss, prog_bar=True)

        return loss
