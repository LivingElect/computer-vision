import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from datasets.voc_dataset import VOCDataset, my_collate_fn
from models.my_unet import UNet
from utils.metrics import compute_ciou_loss

import os # 记得在文件开头导入 os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 炼丹炉已点火，当前使用计算引擎: {device}")

    # 1. 图像变换工具箱
    transform = T.Compose([
        T.Resize((256, 256)), # 强行缩放图片为 256x256
        T.ToTensor()
    ])

    dataset = VOCDataset(root_dir='./data/VOCdevkit/VOC2012', yaml_path='./configs/voc.yaml')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn)

    # 2. 召唤真正的双头检测网络！
    model = UNet(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 🌟 核心改造 1：准备两把皮鞭！
    criterion_cls = nn.CrossEntropyLoss() # 惩罚分类猜错
    criterion_box = nn.L1Loss()           # 惩罚坐标画歪 (计算绝对值误差)

    print("\n🚀 解开封印，突入真正的目标检测死循环！")
    
    for epoch in range(5): 
        model.train() 
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            
            # --------------------------------------------------
            # 🧱 动作 0：数据清洗、降维与【坐标同步缩放】
            # --------------------------------------------------
            # A. 转换图片并压入显卡
            tensor_images = torch.stack([transform(img) for img in images]).to(device)
            
            # B. 极其严谨地提取“类别”和“坐标”
            simple_labels = []
            simple_boxes = []
            
            for i, t in enumerate(targets):
                # 🌟 核心改造 2：坐标必须跟着图片一起缩放！
                # 拿到这第一张图在 Resize 之前的原始宽和高
                orig_w, orig_h = images[i].size 
                scale_x = 256.0 / orig_w  # 算出 X 轴缩放比例
                scale_y = 256.0 / orig_h  # 算出 Y 轴缩放比例

                if len(t['labels']) > 0:
                    # 抓取类别
                    simple_labels.append(t['labels'][0])
                    
                    # 抓取原始坐标 [xmin, ymin, xmax, ymax] 并乘以缩放比例！
                    orig_box = t['boxes'][0]
                    scaled_box = torch.tensor([
                        orig_box[0] * scale_x,
                        orig_box[1] * scale_y,
                        orig_box[2] * scale_x,
                        orig_box[3] * scale_y
                    ], dtype=torch.float32)
                    simple_boxes.append(scaled_box)
                else:
                    simple_labels.append(torch.tensor(0))
                    simple_boxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))
            
            # 叠成张量压入显卡！
            tensor_labels = torch.stack(simple_labels).to(device)
            tensor_boxes = torch.stack(simple_boxes).to(device)

            # --------------------------------------------------
            # 🔪 核心杀招：真实检测的双重神圣三连击！
            # --------------------------------------------------
            
            optimizer.zero_grad() 
            
            # 🌟 核心改造 3：网络同时吐出分类结果和框坐标！
            # (前提是你已经按上一节改了 my_unet.py，让它 return cls_out, box_out)
            cls_preds, box_preds = model(tensor_images)              
            
            # 🌟 核心改造 4：双重审判！
            loss_cls = criterion_cls(cls_preds, tensor_labels)   # 分类错了吗？
            # loss_box = criterion_box(box_preds, tensor_boxes)    # 框歪了多少个像素？
            loss_box = compute_ciou_loss(box_preds, tensor_boxes)

            # 🌟 核心改造 5：终极融合！
            # 我们还可以给 loss_box 加个权重，比如告诉网络：“画框比分类更重要，权重乘 2！”
            total_loss = loss_cls + (loss_box * 2.0)
            
            total_loss.backward()                      
            optimizer.step()                     
            
            # 每 100 个 Batch 打印一次极其清晰的双头 Loss
            if batch_idx % 100 == 0: 
                print(f"📈 Epoch [{epoch+1}/5] | Batch [{batch_idx}/{len(dataloader)}] "
                      f"| 分类Loss: {loss_cls.item():.4f} | 坐标误差(像素): {loss_box.item():.2f}")

    print("\n🎉 你的 4070S 已经真正学会了如何在图片上画框！")
    print("\n🎉 训练结束！正在固化网络记忆...")
    
    # 1. 确保存档文件夹存在 (防止因为没建文件夹而报错)
    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 已自动创建存档目录: {save_dir}")
        
    # 2. 组装存档路径
    save_path = os.path.join(save_dir, 'best_model.pth')
    
    # 3. 极其神圣的一步：把 4070S 显存里被千锤百炼的参数，写死到硬盘上！
    torch.save(model.state_dict(), save_path)
    print(f"💾 绝世金丹已出炉，权重文件保存在: {save_path}")

    
if __name__ == '__main__':
    main()



# ...
