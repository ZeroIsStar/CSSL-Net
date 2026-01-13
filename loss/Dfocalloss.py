import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        """
        初始化 DynamicFocalLoss。

        Args:
            gamma (float): 聚焦因子，用于调整难以分类样本的权重。
            reduction (str): 损失的归约方法，可以是'mean', 'sum'或'none'。
        """
        super(DynamicFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        计算 DynamicFocalLoss。

        Args:
            logits (Tensor): 模型的输出，形状为[B, C, H, W]。
            labels (Tensor): 标签，形状为[B, H, W]。

        Returns:
            loss (Tensor): 计算出的 DynamicFocalLoss。
        """
        # 获取类别数量
        num_classes = logits.size(1)

        # 转换标签为 one-hot 编码
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 计算每个类别的样本数量
        class_counts = labels_one_hot.sum(dim=(0, 2, 3))

        # 计算每个类别的权重 alpha
        alpha = 1.0 / (class_counts + 1e-6)
        alpha = alpha / alpha.sum()  # 归一化，使权重和为1

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # 计算概率
        pt = torch.exp(-ce_loss)

        # 根据类别应用 alpha 权重
        alpha_t = alpha.gather(0, labels.view(-1)).view_as(labels)
        # print(alpha_t)

        # 计算 Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 示例使用
if __name__ == "__main__":
    # 创建一个 DynamicFocalLoss 对象
    criterion = DynamicFocalLoss(gamma=2, reduction='mean')

    # 示例 logits 和 labels
    logits = torch.randn(8, 2, 256, 256)  # [B, C, H, W]
    labels = torch.randint(0, 2, (8, 256, 256))  # [B, H, W]

    # 计算损失
    loss = criterion(logits, labels)
    print(loss)
