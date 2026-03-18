import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import yaml

# ==========================================
# 🌟 大厂级核心算子：网格目标生成器
# ==========================================
def build_target_tensor(gt_boxes, gt_classes, grid_size=32, img_size=256):
    """
    YOLOv1 核心：将真实框映射到 32x32 的网格中
    gt_boxes: 这张图上所有真实目标的缩放后坐标 [[xmin, ymin, xmax, ymax], ...]
    gt_classes: 这些目标对应的类别索引 [cls_id1, cls_id2, ...]
    """
    # 初始化全黑审判矩阵，增加到 6 维: (xmin, ymin, xmax, ymax, confidence, class_id)
    target = torch.zeros((grid_size, grid_size, 6))
    
    cell_w = img_size / grid_size  # 每个格子 8 像素宽
    cell_h = img_size / grid_size
    
    for box, cls_id in zip(gt_boxes, gt_classes):
        xmin, ymin, xmax, ymax = box
        
        # 1. 计算中心点
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        # 2. 计算中心点落入的网格索引
        col = int(cx / cell_w)
        row = int(cy / cell_h)
        
        # 防止极端情况越界
        col = min(max(col, 0), grid_size - 1)
        row = min(max(row, 0), grid_size - 1)
        
        # 3. 册封天选之子！
        target[row, col, 4] = 1.0        # 置信度设为 1 (正样本)
        # 🌟 修改这行：将绝对像素 (如 200) 变成相对比例 (如 0.78)
        # 这样 MSE 算出来的误差永远在 0~1 之间，梯度彻底平稳！
        target[row, col, 0] = xmin / img_size
        target[row, col, 1] = ymin / img_size
        target[row, col, 2] = xmax / img_size
        target[row, col, 3] = ymax / img_size

        target[row, col, 5] = cls_id     # 告诉这个格子，你要识别的是什么车
        
    # 4. 降维重组：把 32x32 的网格展平，变成 1024 个一维排列的框
    # 这样可以和模型吐出的 [Batch, 1024, X] 完美对齐，方便算 Loss
    target = target.view(-1, 6) # 输出形状: [1024, 6]
    
    return target


class VOCDataset(Dataset):
    def __init__(self, root_dir, yaml_path):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.xml_dir = os.path.join(root_dir, 'Annotations')
        
        # 解决 Windows 编码报错
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.classes = data['names']
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        # 统一缩放尺寸
        self.img_size = 256
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. 读取图片
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 记录原始尺寸用于计算缩放比例
        orig_w, orig_h = image.size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        
        image_tensor = self.transform(image)
        
        # 2. 读取并解析 XML
        xml_name = img_name.replace('.jpg', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        gt_boxes = []
        gt_classes = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_id:
                continue
            
            bndbox = obj.find('bndbox')
            # 乘以缩放比例，把真实坐标也压缩到 256x256 尺度下
            xmin = float(bndbox.find('xmin').text) * scale_x
            ymin = float(bndbox.find('ymin').text) * scale_y
            xmax = float(bndbox.find('xmax').text) * scale_x
            ymax = float(bndbox.find('ymax').text) * scale_y
            
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_classes.append(self.class_to_id[name])
            
        # 3. 🌟 召唤核心算子：生成 1024 维度的审判矩阵
        # 如果这张图里什么都没找到（背景图），传入空列表也不会报错
        target_tensor = build_target_tensor(gt_boxes, gt_classes, grid_size=32, img_size=self.img_size)

        # 直接交出完美的图片张量和目标张量！
        return image_tensor, target_tensor

# ❌ 这里删除了原来那个难懂的 my_collate_fn！
# 因为现在 target_tensor 的形状永远是 [1024, 6]，大小完全一致，不再需要它了。