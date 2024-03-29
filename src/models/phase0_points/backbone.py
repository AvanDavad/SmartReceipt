import torch.nn as nn


class Phase0PointsBackbone(nn.Module):
    def __init__(self, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(
            3, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_2 = nn.Conv2d(
            8, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_3 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_123 = nn.BatchNorm2d(8)

        self.conv_4 = nn.Conv2d(
            8, 8, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_5 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_6 = nn.Conv2d(
            8, 8, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_456 = nn.BatchNorm2d(8)

        self.conv_7 = nn.Conv2d(
            8, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_8 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_9 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_789 = nn.BatchNorm2d(16)

        self.conv_10 = nn.Conv2d(
            16, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_11 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_12 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_101112 = nn.BatchNorm2d(16)

        self.conv_13 = nn.Conv2d(
            16, 16, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_14 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_15 = nn.Conv2d(
            16, 16, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_131415 = nn.BatchNorm2d(16)

        self.conv_16 = nn.Conv2d(
            16, 32, kernel_size=5, stride=1, padding="same", bias=True
        )
        self.conv_17 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_18 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_161718 = nn.BatchNorm2d(32)

        self.conv_19 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_20 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_21 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_192021 = nn.BatchNorm2d(32)

        self.conv_22 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_23 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_24 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_222324 = nn.BatchNorm2d(32)

        self.conv_25 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_26 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv_27 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="same", bias=True
        )
        if self.use_bn:
            self.bn_252627 = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding="valid", bias=True
        )

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = x0 = self.conv_1(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = self.conv_2(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = self.conv_3(x)  # 768x768x8
        x = self.relu(x)  # 768x768x8
        x = self.dropout(x + x0)  # 768x768x8
        if self.use_bn:
            x = self.bn_123(x)  # 768x768x8

        x = x0 = self.avg_pool(x)  # 384x384x8
        x = self.conv_4(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = self.conv_5(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = self.conv_6(x)  # 384x384x8
        x = self.relu(x)  # 384x384x8
        x = self.dropout(x + x0)  # 384x384x8
        if self.use_bn:
            x = self.bn_456(x)  # 384x384x8

        x = self.avg_pool(x)  # 192x192x8
        x = x0 = self.conv_7(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = self.conv_8(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = self.conv_9(x)  # 192x192x16
        x = self.relu(x)  # 192x192x16
        x = self.dropout(x + x0)  # 192x192x16
        if self.use_bn:
            x = self.bn_789(x)  # 192x192x16

        x = x0 = self.avg_pool(x)  # 96x96x16
        x = self.conv_10(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = self.conv_11(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = self.conv_12(x)  # 96x96x16
        x = self.relu(x)  # 96x96x16
        x = self.dropout(x + x0)  # 96x96x16
        if self.use_bn:
            x = self.bn_101112(x)  # 96x96x16

        x = x0 = self.avg_pool(x)  # 48x48x16
        x = self.conv_13(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = self.conv_14(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = self.conv_15(x)  # 48x48x16
        x = self.relu(x)  # 48x48x16
        x = self.dropout(x + x0)  # 48x48x16
        if self.use_bn:
            x = self.bn_131415(x)  # 48x48x16

        x = self.avg_pool(x)  # 24x24x16
        x = x0 = self.conv_16(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = self.conv_17(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = self.conv_18(x)  # 24x24x32
        x = self.relu(x)  # 24x24x32
        x = self.dropout(x + x0)  # 24x24x32
        if self.use_bn:
            x = self.bn_161718(x)  # 24x24x32

        x = x0 = self.avg_pool(x)  # 12x12x32
        x = self.conv_19(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = self.conv_20(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = self.conv_21(x)  # 12x12x32
        x = self.relu(x)  # 12x12x32
        x = self.dropout(x + x0)  # 12x12x32
        if self.use_bn:
            x = self.bn_192021(x)  # 12x12x32

        x = x0 = self.avg_pool(x)  # 6x6x32
        x = self.conv_22(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = self.conv_23(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = self.conv_24(x)  # 6x6x32
        x = self.relu(x)  # 6x6x32
        x = self.dropout(x + x0)  # 6x6x32
        if self.use_bn:
            x = self.bn_222324(x)  # 6x6x32

        x = x0 = self.avg_pool(x)  # 3x3x32
        x = self.conv_25(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = self.conv_26(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = self.conv_27(x)  # 3x3x32
        x = self.relu(x)  # 3x3x32
        x = self.dropout(x + x0)  # 3x3x32
        if self.use_bn:
            x = self.bn_252627(x)  # 3x3x32

        x = self.final_conv(x)  # 1x1x32

        return x
