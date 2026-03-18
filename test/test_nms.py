import torch
# 导入我们刚写的兵器
from utils.metrics import nms

def run_nms_test():
    print("🔥 NMS 擂台赛模拟开始...")
    
    # 模拟 3 个候选框 [xmin, ymin, xmax, ymax]
    # 框0 和 框1 极其接近 (都框住了同一辆车)
    # 框2 离得很远 (框住了一个行人)
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],  # 框 0 (车-很准)
        [12.0, 12.0, 52.0, 52.0],  # 框 1 (车-稍微歪一点)
        [100.0, 100.0, 150.0, 150.0] # 框 2 (行人)
    ])
    
    # 模拟这 3 个框的自信度得分
    scores = torch.tensor([0.9, 0.8, 0.7])
    
    print("\n📦 原始输入:")
    for i in range(3):
        print(f"框 {i}: 得分 {scores[i]:.2f}, 坐标 {boxes[i].tolist()}")

    # 执行大清洗 (阈值设为 0.5)
    iou_thresh = 0.5
    print(f"\n🔪 执行 NMS... (生死阈值 = {iou_thresh})")
    survivors = nms(boxes, scores, iou_threshold=iou_thresh)
    
    print(f"\n🎉 清场结束！存活下来的框的索引是: {survivors}")
    print("存活的实体框信息:")
    for idx in survivors:
         print(f"框 {idx}: 得分 {scores[idx]:.2f}, 坐标 {boxes[idx].tolist()}")

if __name__ == "__main__":
    run_nms_test()