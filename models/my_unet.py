import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    # ... (保持原来的 DoubleConv 不变) ...
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=20):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # ❌ 删除了极其暴力的 AdaptiveAvgPool2d 和 Flatten
        
        # 🌟 换成 1x1 卷积头！它能在不破坏图像空间结构的情况下，改变通道数。
        # 头 A：分类头
        self.cls_head = nn.Conv2d(512, num_classes, kernel_size=1) 
        
        # 头 B：回归头
        self.box_head = nn.Conv2d(512, 4, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        
        # 此时 x4 的形状大概是 [Batch, 512, 32, 32] (假设输入是 256x256)
        
        # 吐出密集预测结果
        cls_preds = self.cls_head(x4) # 形状: [Batch, 20, 32, 32]
        box_preds = self.box_head(x4) # 形状: [Batch, 4, 32, 32]
        
        # 🔄 极其关键的“降维重组”：把 32x32 的网格展平，变成 1024 个候选框
        # [Batch, 20, 32, 32] -> [Batch, 20, 1024] -> [Batch, 1024, 20]
        cls_preds = cls_preds.flatten(2).permute(0, 2, 1) 
        box_preds = box_preds.flatten(2).permute(0, 2, 1)
        
        # 用 Sigmoid 强行把网络吐出的坐标死死压在 0 到 1 之间！绝对不允许越界爆炸！
        box_preds = torch.sigmoid(box_preds)
        
        # 最终交出去的数据结构：1024 个框，每个框有 20 个类别的打分和 4 个坐标
        return cls_preds, box_preds