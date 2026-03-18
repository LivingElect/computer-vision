import torch
import torch.nn as nn
import torch.nn.functional as F

print("⚖️ 神经网络刑场：交叉熵损失 (Cross Entropy) 手撕现场\n")

# 1. 案发现场模拟
# 假设我们有 3 个类别 (0:猫, 1:狗, 2:车)
# 网络吐出了 3 个极其随意的实数 (Logits)，它显然觉得这玩意最可能是 "狗" (因为 3.2 最大)
logits = torch.tensor([[1.5, 3.2, -0.5]])  # 形状 [1, 3]

# 但真实的标签 (Ground Truth) 其实是 "猫" (类别索引为 0)
target = torch.tensor([0])                 # 形状 [1]

print(f"🤐 网络原始输出 (Logits): {logits.tolist()}")
print(f"🎯 真实正确答案: 类别 {target.item()} (猫)")

# ==========================================
# 🔪 方式一：纯手工数学推导 (剥开黑盒！)
# ==========================================
# 步骤 A：用 Softmax 把实数变成概率 (加起来等于 1)
# 公式：exp(x_i) / sum(exp(x_j))
probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
print(f"\n📊 转化为概率分布: {probs.tolist()}")
print(f"   (网络只有 {(probs[0, 0].item() * 100):.1f}% 的自信认为它是猫)")

# 步骤 B：求自然对数
log_probs = torch.log(probs)

# 步骤 C：极其残忍的 Negative Log Likelihood (NLL)
# 只抽出正确答案(类别 0)对应的那个对数值，然后加个负号！
manual_loss = -log_probs[0, target[0].item()]
print(f"💥 纯手工计算出的 Loss 惩罚值: {manual_loss.item():.4f}")

# ==========================================
# 🛡️ 方式二：调用 PyTorch 官方 API 对账
# ==========================================
criterion = nn.CrossEntropyLoss()
official_loss = criterion(logits, target)
print(f"✅ PyTorch 官方计算的 Loss: {official_loss.item():.4f}")

if abs(manual_loss - official_loss) < 1e-5:
    print("\n🎉 完美对齐！你已经彻底掌握了交叉熵的底层数学灵魂！")