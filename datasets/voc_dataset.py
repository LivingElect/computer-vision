import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
import yaml

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, yaml_path):
        # 1. 明确两大仓库的位置
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.xml_dir = os.path.join(root_dir, 'Annotations')
        
        # 2. 读取 YAML 里的类别字典 (比如 'car': 6)
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # 构建 name 到 ID 的映射字典
            self.class_to_id = {name: idx for idx, name in enumerate(data['names'])}
            
        # 3. 拿到所有图片的名字列表 (去掉后缀)
        self.img_names = [f.split('.')[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        # 告诉 DataLoader，我们一共有多少张图
        return len(self.img_names)

    # ==========================================
    # 🌟 核心替换区：这里就是你刚才看到的“手撕 XML”逻辑的实战应用！
    # DataLoader 每次要数据，都会疯狂调用这个函数
    # ==========================================
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # 1. 极其优雅的路径拼接
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        xml_path = os.path.join(self.xml_dir, img_name + '.xml')
        
        # 2. 读图 (保持 PIL 格式，缩放留给 train.py 里的 transform)
        image = Image.open(img_path).convert("RGB")
        
        # --------------------------------------------------
        # 👇 下面这块，就是我刚才让你必须“白板手撕”的 XML 剥洋葱代码！
        # --------------------------------------------------
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            label_id = self.class_to_id[name] # 查字典，变数字
            
            bndbox = obj.find('bndbox')
            # ⚠️ 极其致命的细节：字符串强转 float
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)
        # --------------------------------------------------
        # 👆 XML 解析结束
        # --------------------------------------------------
        
        # 3. 把剥出来的数字变成 PyTorch 的张量
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        # 4. 把原始图片和装满坐标的字典一起交出去！
        return image, target

# ==========================================
# 防崩溃补丁 (必须顶格)
# ==========================================
def my_collate_fn(batch):
    return tuple(zip(*batch))