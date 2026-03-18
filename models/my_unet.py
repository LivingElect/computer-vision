import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(卷积 => 批归一化 => ReLU) * 2次，提取特征的经典小模块"""
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
        
        # ==========================================
        # 1. 骨干网络 (Backbone) - 疯狂提取特征
        # ==========================================
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # 🌟 降维打击：全局平均池化。
        # 不管输入的特征图有多大，强行压缩成 1x1，将其变成“单目标检测器”
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==========================================
        # 2. 解耦头 (Decoupled Heads) - 大厂标配
        # ==========================================
        # 头 A：专门猜这是什么车 (分类)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 训练时随机断开神经元，防止死记硬背
            nn.Linear(256, num_classes) # 输出形状: [Batch, 20]
        )
        
        # 头 B：专门猜这辆车在哪 (坐标框回归)
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4), # 输出形状: [Batch, 4] 分别是 xmin, ymin, xmax, ymax
            nn.ReLU() # 坐标不能是负数，加个 ReLU 兜底保平安
        )

    def forward(self, x):
        # 1. 图像进入骨干网络
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. 压缩成高维特征向量 [Batch, 512, 1, 1]
        feat = self.pool(x4)
        
        # 3. 特征向量分别流向两个不同的“脑区”
        cls_preds = self.cls_head(feat) # 吐出分类打分
        box_preds = self.box_head(feat) # 吐出坐标数值
        
        # 👑 极其致命的一行：必须且只能交出这两个变量！(解决你报错的罪魁祸首)
        return cls_preds, box_preds

# ==========================================
# 本地测试代码 (仅用于验证模型有没有写错，不会在 train.py 中运行)
# ==========================================
if __name__ == '__main__':
    # 模拟一张 256x256 的伪造图片送入网络
    dummy_input = torch.randn(2, 3, 256, 256)
    model = UNet(num_classes=20)
    
    cls_out, box_out = model(dummy_input)
    print("模型测试成功！")
    print(f"分类头输出形状: {cls_out.shape} -> 期望是 [2, 20]")
    print(f"回归头输出形状: {box_out.shape} -> 期望是 [2, 4]")