import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os

# 导入你亲手打造的兵器
from models.my_unet import UNet
from utils.metrics import nms

import yaml # 导入 yaml 库

def get_classes_from_yaml(yaml_path):
    """
    从配置文件中动态提取类别名单
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names'] # 返回 YAML 里的列表

# 在 predict() 函数里，把硬编码的列表换成这一行：
VOC_CLASSES = get_classes_from_yaml('./configs/voc.yaml')


def predict():
    # 1. 环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = './checkpoints/best_model.pth'
    img_path = './test/test_image.jpg'

    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：找不到权重文件 {checkpoint_path}，请先运行 train.py")
        return

    # 2. 加载模型与权重
    print(f"🧠 正在加载模型至 {device}...")
    model = UNet(num_classes=20).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 🌟 极其重要：切换到推理模式！

    # 3. 图像预处理
    print(f"📸 正在处理图片: {img_path}")
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"❌ 错误：无法读取图片 {img_path}")
        return
    
    orig_h, orig_w = raw_img.shape[:2]
    # BGR 转 RGB 并转为 Tensor
    pil_img = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device) # 增加 Batch 维度

   # ... (前面的加载模型和读取图片代码保持不变) ...

    # 4. 前向传播 (此时网络会吐出 1024 个框)
    with torch.no_grad():
        cls_logits, box_preds = model(input_tensor)
        
        # 取出单张图片的结果: [1024, 20] 和 [1024, 4]
        cls_preds_single = cls_logits[0]
        box_preds_single = box_preds[0]
        
        # 把分类 Logit 变成概率
        probs = torch.sigmoid(cls_preds_single) 
        max_scores, class_indices = torch.max(probs, dim=-1)

    # 5. 🌟 真正的 NMS 大清洗开始！
    
    # 步骤 A：粗筛 (置信度过滤)。先把得分低于 0.3 的垃圾框直接物理抹杀
    score_threshold = 0.3
    mask = max_scores > score_threshold
    
    surviving_boxes = box_preds_single[mask]     # 活下来的坐标
    surviving_scores = max_scores[mask]          # 活下来的分数
    surviving_classes = class_indices[mask]      # 活下来的类别
    
    print(f"🧹 粗筛后，1024 个框还剩 {len(surviving_boxes)} 个")

    if len(surviving_boxes) == 0:
        print("❌ 没有检测到任何置信度大于 0.3 的目标。")
    else:
        # 步骤 B：坐标还原到原图比例
        scale_x, scale_y = orig_w / 256.0, orig_h / 256.0
        # 广播机制缩放所有活下来的框
        surviving_boxes[:, 0] *= orig_w
        surviving_boxes[:, 1] *= orig_h
        surviving_boxes[:, 2] *= orig_w
        surviving_boxes[:, 3] *= orig_h
        
        # 步骤 C：精筛 (手写 NMS 登场！)，解决重叠问题
        iou_threshold = 0.45
        # ⚠️ 注意：如果你之前的 metrics.py 里的 nms 函数期望的是 CPU tensor，这里加上 .cpu()
        keep_indices = nms(surviving_boxes.cpu(), surviving_scores.cpu(), iou_threshold)
        print(f"🎉 NMS 清场后，最终剩余 {len(keep_indices)} 个真实目标！")

        # 6. OpenCV 渲染结果
        for idx in keep_indices:
            b = surviving_boxes[idx].cpu().numpy().astype(int)
            score = surviving_scores[idx].item()
            label_name = VOC_CLASSES[surviving_classes[idx].item()]
            
            cv2.rectangle(raw_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(raw_img, f"{label_name} {score:.2f}", (b[0], b[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 7. 存盘 (应对 Linux 服务器无界面的情况)
    save_path = "result_output_nms.jpg"
    cv2.imwrite(save_path, raw_img)
    print(f"🚀 结果已成功保存至: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    predict()