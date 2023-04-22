import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class CNNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        import pdb;pdb.set_trace()
        # Load a pretrained ResNet model (e.g., ResNet-18)
        self.backbone = models.resnet18(pretrained=True)

        # Replace the last fully connected layer to match the desired output size
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 12)

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

if __name__ == "__main__":
    model = CNNModule()