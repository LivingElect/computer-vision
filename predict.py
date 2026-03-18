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

    # 4. 前向传播 (核心推理)
    with torch.no_grad(): # 🌟 开启“省电模式”：不计算梯度
        cls_logits, box_preds = model(input_tensor)
        
        # 处理分类结果
        probs = torch.softmax(cls_logits[0], dim=0)
        score, class_idx = torch.max(probs, dim=0)
        label = VOC_CLASSES[class_idx.item()]
        
        # 处理坐标结果 (256x256 尺度下的坐标)
        pred_box = box_preds[0].cpu().numpy()

    # 5. 坐标逆向还原 (从 256 还原到原始图片像素)
    scale_x, scale_y = orig_w / 256.0, orig_h / 256.0
    real_box = [
        pred_box[0] * scale_x, pred_box[1] * scale_y,
        pred_box[2] * scale_x, pred_box[3] * scale_y
    ]

    # 6. 🌟 模拟 NMS 流程
    # 既然是实验，我们围绕预测框生成 5 个伪随机冗余框，测试 NMS 是否生效
    boxes = [real_box]
    scores = [score.item()]
    for _ in range(4): # 制造 4 个干扰项
        noise = np.random.randint(-10, 10, size=4)
        boxes.append([real_box[0]+noise[0], real_box[1]+noise[1], real_box[2]+noise[2], real_box[3]+noise[3]])
        scores.append(score.item() * 0.9) # 干扰项分数稍低

    # 转换成张量喂给 NMS
    t_boxes = torch.tensor(boxes, dtype=torch.float32)
    t_scores = torch.tensor(scores, dtype=torch.float32)
    keep_indices = nms(t_boxes, t_scores, iou_threshold=0.5)

    # 7. OpenCV 渲染结果
    print(f"✅ 检测完成！结果：{label} (置信度: {score.item():.2f})")
    for idx in keep_indices:
        b = t_boxes[idx].numpy().astype(int)
        cv2.rectangle(raw_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(raw_img, f"{label} {scores[idx]:.2f}", (b[0], b[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示结果
    save_path = "result_output.jpg"
    cv2.imwrite(save_path, raw_img)
    print(f"🚀 结果已成功保存至: {os.path.abspath(save_path)}")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict()