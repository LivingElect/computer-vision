# VOC AutoDrive Project

🚀 基于UNet的VOC2012语义分割项目，用于自动驾驶场景中的目标检测与分割。

## 项目简介

本项目实现了基于UNet模型的语义分割系统，针对VOC2012数据集进行训练和评估。主要功能包括：

- 完整的UNet模型实现，包含编码器-解码器结构和跳跃连接
- VOC2012数据集的加载和预处理
- 交叉熵损失函数和IoU评估指标
- 完整的训练、验证和模型保存流程
- 非极大值抑制（NMS）算法实现

## 目录结构

```
VOC_AutoDrive_Project/
 ├── train.py                     # 👑 全局指挥官：负责组装所有模块，启动训练大循环
 ├── predict.py                   # 🎯 预测脚本：用于模型推理和结果可视化
 ├── README.md                    # 📖 项目说明文档
 ├── .gitignore                   # 🚫 Git忽略文件配置
 ├── configs/                     # 🧠 配置大脑：存放所有的超参数和类别定义
 │   └── voc.yaml                 # 定义了20个类别名和数据集基本信息
 ├── data/                        # 📦 重资产仓库：存放动辄几十GB的数据集 (Git不追踪)
 │   └── VOCdevkit/
 │       └── VOC2012/             # 包含Annotations(XML)和JPEGImages(JPG)
 ├── datasets/                    # 🚜 数据榨汁机：负责脏数据到张量(Tensor)的转化
 │   ├── __init__.py              # 声明这是一个Python包
 │   └── voc_dataset.py           # 包含VOCDataset类和my_collate_fn补丁
 ├── models/                      # 🦾 战斗引擎：定义神经网络的层级结构
 │   ├── __init__.py
 │   └── my_unet.py               # 存放UNet类的实现
 ├── utils/                       # 📏 精密量具：存放数学工具和评估指标
 │   ├── __init__.py
 │   └── metrics.py               # IoU (交并比) 核心代码
 ├── test/                        # 🧪 实验场：存放独立的测试脚本
 │   ├── __init__.py
 │   ├── test_loss.py             # 验证交叉熵数学逻辑
 │   └── test_nms.py              # 验证非极大值抑制算法
 └── checkpoints/                 # 💾 存档点：存放训练出来的模型权重
     └── best_model.pth           # 训练得到的最佳模型
```

## 安装依赖

```bash
# 克隆仓库
git clone https://github.com/LivingElect/computer-vision.git
cd computer-vision

# 安装依赖
pip install torch torchvision tqdm pyyaml
```

## 数据集准备

1. 下载VOC2012数据集：
   - 从[官方网站](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)下载VOC2012数据集
   - 解压到`data/VOCdevkit/VOC2012/`目录

2. 数据集结构：
   - `JPEGImages/`：包含原始图像
   - `SegmentationClass/`：包含语义分割标注
   - `ImageSets/Segmentation/`：包含训练和验证集的图像列表

## 训练模型

```bash
# 启动训练
python train.py
```

训练过程中，模型会自动保存到`checkpoints/`目录，文件名为`best_model.pth`。

## 模型预测

```bash
# 运行预测脚本
python predict.py --image_path test/test_image.jpg --model_path checkpoints/best_model.pth
```

预测结果会保存为`result_output.jpg`。

## 测试脚本

```bash
# 测试交叉熵损失函数
python test/test_loss.py

# 测试非极大值抑制算法
python test/test_nms.py
```

## 模型架构

本项目使用UNet模型，其架构特点包括：

- **编码器**：通过一系列卷积和池化操作，提取图像的高级特征
- **解码器**：通过上采样和跳跃连接，恢复图像的空间信息
- **跳跃连接**：将编码器不同层次的特征与解码器对应层次的特征融合，提高分割精度

## 评估指标

使用IoU（交并比）作为主要评估指标，计算公式为：

```
IoU = 交集面积 / 并集面积
```

## 超参数配置

在`configs/voc.yaml`文件中可以修改以下超参数：

- `batch_size`：批量大小
- `lr`：学习率
- `epochs`：训练轮数
- `num_classes`：类别数

## 技术栈

- **框架**：PyTorch
- **模型**：UNet
- **数据集**：VOC2012
- **评估指标**：IoU
- **优化器**：Adam
- **损失函数**：交叉熵损失

## 许可证

本项目采用MIT许可证。

## 致谢

- Pascal VOC数据集团队
- UNet论文作者
- PyTorch社区

## 联系方式

如果您有任何问题或建议，请随时联系我们。
