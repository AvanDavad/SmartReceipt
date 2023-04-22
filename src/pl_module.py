import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class CNNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet model (e.g., ResNet-18)
        self.backbone = models.resnet18(pretrained=True)

        # Replace the last fully connected layer to match the desired output size
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 12)
        self.loss_module = nn.L1Loss()

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
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

if __name__ == "__main__":
    model = CNNModule()