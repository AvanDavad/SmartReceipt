import torch.nn as nn

from src.datasets.phase1line_dataset import Phase1LineDataset


class Phase1LineBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(
            12, 8, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.bn_1 = nn.BatchNorm2d(8)

        self.conv_2 = nn.Conv2d(
            8, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.bn_2 = nn.BatchNorm2d(16)

        self.conv_3 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.bn_3 = nn.BatchNorm2d(16)

        self.conv_4 = nn.Conv2d(
            16, 24, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.bn_4 = nn.BatchNorm2d(24)

        self.conv_5 = nn.Conv2d(
            24, 24, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.bn_5 = nn.BatchNorm2d(24)

        self.conv_6 = nn.Conv2d(
            24, 32, kernel_size=3, stride=1, padding="valid", bias=True
        )
        self.bn_6 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        assert Phase1LineDataset.IMG_SIZE == 100
        assert x.shape[1:] == (12, 100, 100)

        x = self.conv_1(x)  # 100x100 channels: 8
        x = self.relu(x)  # 100x100 channels: 8
        x = self.bn_1(x)  # 100x100 channels: 8
        x = self.dropout(x)  # 100x100 channels: 8

        x = self.avg_pool(x)  # 50x50 channels: 8
        x = self.conv_2(x)  # 50x50 channels: 16
        x = self.relu(x)  # 50x50 channels: 16
        x = self.bn_2(x)  # 50x50 channels: 16
        x = self.dropout(x)  # 50x50 channels: 16

        x = self.avg_pool(x)  # 25x25 channels: 16
        x = self.conv_3(x)  # 25x25 channels: 16
        x = self.relu(x)  # 25x25 channels: 16
        x = self.bn_3(x)  # 25x25 channels: 16
        x = self.dropout(x)  # 25x25 channels: 16

        x = self.avg_pool(x)  # 12x12 channels: 16
        x = self.conv_4(x)  # 12x12 channels: 24
        x = self.relu(x)  # 12x12 channels: 24
        x = self.bn_4(x)  # 12x12 channels: 24
        x = self.dropout(x)  # 12x12 channels: 24

        x = self.avg_pool(x)  # 6x6 channels: 24
        x = self.conv_5(x)  # 6x6 channels: 24
        x = self.relu(x)  # 6x6 channels: 24
        x = self.bn_5(x)  # 6x6 channels: 24
        x = self.dropout(x)  # 6x6 channels: 24

        x = self.avg_pool(x)  # 3x3 channels: 24
        x = self.conv_6(x)  # 1x1 channels: 32

        return x
