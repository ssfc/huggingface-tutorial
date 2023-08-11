import torch
import torch.nn.functional as F

# 模型输出和真实标签
logits = torch.tensor([[1.0, 2.0, 3.0]])
labels = torch.tensor([2])  # 正确标签是第3个类别

# 计算交叉熵损失
loss = F.cross_entropy(logits, labels)

# 计算梯度
loss.backward()

print(loss.item())  # 输出损失的数值
