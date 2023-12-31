from typing import Dict, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.datasets.phase2_single_char_dataset import Phase2SingleCharDataset
from src.datasets.phase2char_dataset import ALL_CHARS
from torch import Tensor
from PIL import Image


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2, bias=True)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_final = nn.Conv2d(128, 128, kernel_size=2, padding="valid", bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        assert Phase2SingleCharDataset.IMG_SIZE == 128
        assert x.shape == (bs, 3, 128, 128)

        x = self.conv_1(x) # 64x64 channels: 8

        x = self.avg_pool(x) # 32x32 channels: 8
        x = x0 = self.conv_2(x) # 32x32 channels: 16
        x = self.relu(x) # 32x32 channels: 16
        x = x0 + x # 32x32 channels: 16

        x = self.avg_pool(x) # 16x16 channels: 16
        x = x0 = self.conv_3(x) # 16x16 channels: 32
        x = self.relu(x) # 16x16 channels: 32
        x = x0 + x # 16x16 channels: 32

        x = self.avg_pool(x) # 8x8 channels: 32
        x = x0 = self.conv_4(x) # 8x8 channels: 64
        x = self.relu(x) # 8x8 channels: 64
        x = x0 + x # 8x8 channels: 64

        x = self.avg_pool(x) # 4x4 channels: 64
        x = x0 = self.conv_5(x) # 4x4 channels: 128
        x = self.relu(x) # 4x4 channels: 128
        x = x0 + x # 4x4 channels: 128

        x = self.avg_pool(x) # 2x2 channels: 128
        x = self.conv_final(x) # 1x1 channels: 128

        return x

class CNNModulePhase2SingleChar(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = len(ALL_CHARS) + 1

        self.cnn_backbone = CNNBackbone()
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(128, self.num_classes)

        self.ce = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["img"]

        image_features = self.cnn_backbone(images)
        image_features = self.flatten(image_features)

        logits = self.linear(image_features)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)

        loss = self.ce(logits, batch["target"].flatten())
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)

        loss = self.ce(logits, batch["target"].flatten())
        self.log("val_loss", loss)
        self.validation_step_outputs.append(loss)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        print(f"\n\nepoch {self.current_epoch}: train_loss={avg_loss}")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        print(f"\n\nepoch {self.current_epoch}: val_loss={avg_loss}")
        self.validation_step_outputs.clear()

    def inference(self, char_image: Image.Image) -> Tuple[str, float]:

        side_len = max(char_image.width, char_image.height)
        square_image = Image.new("RGB", (side_len, side_len), color=(255, 255, 255))
        square_image.paste(char_image, (int((side_len - char_image.width) / 2), int((side_len - char_image.height) / 2)))

        input_tensor = Phase2SingleCharDataset.TRANSFORMS(square_image).unsqueeze(0)
        input_batch = {"img": input_tensor}

        logits = self.forward(input_batch)

        probs = torch.softmax(logits, dim=1)[0]
        prob, idx = probs.max(dim=0)

        if idx == len(ALL_CHARS):
            predicted_char = "unknown"
        else:
            predicted_char = ALL_CHARS[idx]

        return predicted_char, prob.item()