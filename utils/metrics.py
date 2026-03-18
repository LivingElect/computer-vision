import torch
import math

def compute_iou(box1, box2):
    """
    输入：box1 和 box2 都是一维张量 [xmin, ymin, xmax, ymax]
    (面试时为了简化，通常只让你写一对一的 IoU)
    """
    # 🌟 破局点 1：找交集矩形的“左上角”和“右下角”
    # 左上角取最大值 (往里缩)
    inter_lt = torch.max(box1[:2], box2[:2])  # left-top: [xmin, ymin]
    # 右下角取最小值 (往里缩)
    inter_rb = torch.min(box1[2:], box2[2:])  # right-bottom: [xmax, ymax]

    # 🌟 破局点 2：极其优雅的 clamp 消除 if 语句
    # 如果两个框没相交，右下角坐标减去左上角坐标会变成负数。
    # clamp(min=0) 像一把无情的铡刀，把所有负数瞬间切成 0！
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    
    # 算交集面积
    inter_area = inter_wh[0] * inter_wh[1]

    # 算各自的原始面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 🌟 破局点 3：容斥原理算并集
    union_area = area1 + area2 - inter_area

    return inter_area / union_area
    
def compute_ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    计算 CIoU Loss。全程无 for 循环，极限压榨 GPU 算力！
    pred_boxes: 预测框，形状 [Batch, 4] 格式为 [xmin, ymin, xmax, ymax]
    target_boxes: 真实框，形状 [Batch, 4]
    eps: 极其关键的防爆机制 (防止除以 0 导致 Loss 变成 NaN)
    """
    
    # ==========================================
    # 0. 剥离坐标，准备基础物理量
    # ==========================================
    # 分别抽出预测框和真实框的 x1, y1, x2, y2
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
    
    # 计算宽 (w) 和高 (h)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    # 计算中心点坐标 (cx, cy)
    cx1, cy1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    cx2, cy2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2

    # ==========================================
    # 支柱 1：计算基础 IoU (面积重合度)
    # ==========================================
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # clamp(0) 切掉负数，算交集面积
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    
    iou = inter_area / union_area

    # ==========================================
    # 支柱 2：计算中心点距离惩罚 (Distance)
    # ==========================================
    # 1. 算出两个中心点的欧式距离的平方 (d^2)
    d2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    
    # 2. 算出能把两个框同时包住的“最小外接矩形”
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # 3. 算出外接矩形的对角线长度的平方 (c^2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # ==========================================
    # 支柱 3：计算长宽比惩罚 (Aspect Ratio)
    # ==========================================
    # v 是衡量长宽比一致性的参数。用 arctan 算出角度差，再平方
    # 公式：(4 / pi^2) * (arctan(w2/h2) - arctan(w1/h1))^2
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    
    # alpha 是动态权重：当 IoU 很小时，优先优化面积；当 IoU 很大时，优先优化长宽比
    # ⚠️ 极其关键的工程细节：阻断 alpha 的梯度传播！
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # ==========================================
    # 👑 终极融合：CIoU 公式
    # ==========================================
    ciou = iou - (d2 / c2) - (alpha * v)
    
    # 变成 Loss (越大越差，所以用 1 去减)
    loss = 1 - ciou
    
    return loss.mean()

def compute_iou_for_nms(box, boxes):
    """
    专为 NMS 设计的 IoU 计算（支持 1 对多的广播计算）
    box: [4] 大哥的框
    boxes: [N, 4] 剩下所有小弟的框
    """
    # 找交集的左上角和右下角
    inter_tl = torch.max(box[:2], boxes[:, :2])
    inter_br = torch.min(box[2:], boxes[:, 2:])
    
    # 计算交集面积 (clamp(0) 防止没相交时出现负数)
    inter_wh = (inter_br - inter_tl).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    
    # 计算各自的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 计算并集面积
    union_area = box_area + boxes_area - inter_area
    
    return inter_area / union_area

def nms(boxes, scores, iou_threshold=0.5):
    """
    白板级 NMS 核心实现
    """
    keep = [] # 存活名单
    order = scores.argsort(descending=True) # 按分数降序排列的索引
    
    while order.numel() > 0:
        # 1. 永远保送当前最高分的大哥
        best_idx = order[0].item()
        keep.append(best_idx)
        
        if order.numel() == 1:
            break
            
        # 2. 拿出大哥和剩下的小弟
        best_box = boxes[best_idx]
        other_indices = order[1:]
        other_boxes = boxes[other_indices]
        
        # 3. 计算 IoU
        ious = compute_iou_for_nms(best_box, other_boxes)
        
        # 4. 物理抹杀！只保留 IoU <= 阈值的小弟
        survivor_mask = ious <= iou_threshold
        order = other_indices[survivor_mask] # 更新还活着的索引名单
        
    return keep