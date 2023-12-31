import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights


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
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-6)
        return optimizer

    def _combine_losses(self, kps_loss, cls_loss):
        return 20 * kps_loss + cls_loss

    def training_step(self, batch, batch_idx):
        inp_img, kps, is_last = batch

        kps_pred, is_last_pred = self(inp_img)

        kps_loss = self.kps_loss_module(kps, kps_pred)
        cls_loss = self.bce_loss_module(is_last_pred, is_last)
        loss = self._combine_losses(kps_loss, cls_loss)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inp_img, kps, is_last = batch

        kps_pred, is_last_pred = self(inp_img)

        kps_loss = self.kps_loss_module(kps, kps_pred)
        cls_loss = self.bce_loss_module(is_last_pred, is_last)
        loss = self._combine_losses(kps_loss, cls_loss)

        self.log("val_loss", loss, prog_bar=True)

        return loss
