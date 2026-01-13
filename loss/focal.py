import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 首先，我们需要将inputs的形状从[batch_size, channels, H, W]转换为[batch_size, H, W, channels]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        # 将inputs变形为[batch_size * H * W, channels]
        inputs = inputs.view(-1, inputs.shape[-1])

        # 将targets也变形为[batch_size * H * W]
        targets = targets.view(-1)

        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)

        # 选取每个真实标签对应的概率
        class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()

        # 计算focal loss的系数
        focal_weight = torch.pow(1. - class_probs, self.gamma)

        # 计算交叉熵损失
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算最终的focal loss
        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()


# criterion = FocalLoss()
# logits = torch.randn(8, 2, 256, 256)  # Example logits
# labels = torch.randint(0, 2, (8, 256, 256))  # Example labels
# loss = criterion(logits, labels)
# print(loss)