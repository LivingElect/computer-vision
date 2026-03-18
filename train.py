import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入你亲手打造的兵器
from datasets.voc_dataset import VOCDataset
from models.my_unet import UNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 炼丹炉已点火，当前使用计算引擎: {device}")

    # ==========================================
    # 1. 数据集加载 (极其清爽！)
    # ==========================================
    dataset = VOCDataset(root_dir='./data/VOCdevkit/VOC2012', yaml_path='./configs/voc.yaml')
    # 你的 i7-14700 极其强悍，开启 num_workers=4 压榨 CPU 帮忙搬砖
    # drop_last=True 防止最后几张图凑不够一个 Batch 导致报错
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    model = UNet(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 2. 损失函数 (大厂黑科技)
    # ==========================================
    box_criterion = nn.MSELoss() # 坐标用均方误差
    
    # 🌟 极其致命的改动：用 BCE 替代普通的交叉熵！
    # 为什么？因为 BCE 可以同时处理多标签。
    # 对于正样本，我们让它逼近真实类别的 1。对于负样本(背景)，我们逼它把所有 20 个类的打分全降到 0！
    cls_criterion = nn.BCEWithLogitsLoss() 

    epochs = 5
    print("🚀 解开封印，突入 1024 维度的密集预测死循环！")

    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device) # 形状: [Batch, 1024, 6]

            optimizer.zero_grad()

            # 前向传播
            cls_preds, box_preds = model(images) # [B, 1024, 20], [B, 1024, 4]

            # ==========================================
            # 3. 🌟 核心：大厂级正负样本 Loss 计算
            # ==========================================
            
            # 提取置信度掩码 (1 表示这里有车，0 表示这里是背景)
            # targets[..., 4] 就是我们之前在 Dataset 里赋的 1.0
            obj_mask = targets[..., 4] == 1.0 # 形状: [Batch, 1024]
            
            # --- A. 坐标 Loss (只惩罚天选之子) ---
            pos_box_preds = box_preds[obj_mask]           # 揪出网络预测的正样本坐标
            pos_target_boxes = targets[obj_mask][:, 0:4]  # 揪出真实的坐标
            
            # 如果这批图片里有目标，才算坐标误差
            if pos_box_preds.numel() > 0:
                loss_box = box_criterion(pos_box_preds, pos_target_boxes)
            else:
                loss_box = torch.tensor(0.0, device=device)

            # --- B. 分类与置信度 Loss (全局连坐惩罚) ---
            # 构造一个全 0 的目标类别矩阵 [Batch, 1024, 20]
            target_cls = torch.zeros_like(cls_preds, device=device)
            
            # 只有正样本的地方，对应类别的概率才被强行设为 1.0 (One-Hot 编码)
            if pos_box_preds.numel() > 0:
                pos_class_ids = targets[obj_mask][:, 5].long()
                target_cls[obj_mask, pos_class_ids] = 1.0
            
            # 全局对撞！
            # 天选之子会努力把正确类别的打分推向 1。
            # 陪跑的背景，会因为 target_cls 对应位置全是 0，而遭到灭顶之灾，乖乖把分数降成 0！
            loss_cls = cls_criterion(cls_preds, target_cls)

            # --- C. 算力融合 ---
            # 坐标比分类难学得多，大厂通常会给坐标误差加一个放大系数 (比如乘以 5.0)
            loss = loss_cls + 5.0 * loss_box
            
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"📈 Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | "
                      f"分类(含背景)Loss: {loss_cls.item():.4f} | 坐标Loss: {loss_box.item():.4f}")

    # 4. 固化记忆
    save_dir = './checkpoints'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 满血版 1024 维度金丹已出炉，权重保存在: {save_path}")

if __name__ == '__main__':
    main()