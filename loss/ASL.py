import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSegmentationLoss(nn.Module):
    def __init__(self, num_classes, initial_weights=None, temperature=1.0,
                 eps=1e-8, dynamic_focus=True):
        """
        自适应语义分割损失函数

        参数:
            num_classes: 类别数量
            initial_weights: 各类别初始权重 (None则为均匀权重)
            temperature: 权重调整的温度参数
            dynamic_focus: 是否启用动态难例挖掘
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps
        self.dynamic_focus = dynamic_focus

        if initial_weights is None:
            self.weights = nn.Parameter(torch.ones(num_classes) / num_classes,
                                        requires_grad=False)
        else:
            self.weights = nn.Parameter(initial_weights.float(),
                                        requires_grad=False)

        # 注册缓冲区用于跟踪类别统计信息
        self.register_buffer('class_hist', torch.zeros(num_classes))
        self.register_buffer('update_counter', torch.zeros(1))

    def forward(self, pred, target):
        # 计算基础交叉熵损失
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # 计算每个类别的出现频率
        one_hot_target = F.one_hot(target, self.num_classes).float()
        class_counts = one_hot_target.sum(dim=(1, 2))  # [B, C] -> [C]

        # 更新类别统计信息 (指数移动平均)
        if self.training:
            current_counts = class_counts.sum(dim=0)
            self.class_hist = 0.9 * self.class_hist + 0.1 * current_counts
            self.update_counter += 1

        # 计算类别权重 (频率的倒数)
        norm_class_hist = self.class_hist / (self.class_hist.sum() + self.eps)
        class_weights = 1.0 / (norm_class_hist + self.eps)
        class_weights = class_weights / class_weights.sum() * self.num_classes

        # 温度调节的softmax权重
        adjusted_weights = F.softmax(class_weights / self.temperature, dim=0)

        # 计算加权交叉熵损失
        weighted_ce = ce_loss * adjusted_weights[target]
        weighted_ce = weighted_ce.mean()

        # 动态难例挖掘
        if self.dynamic_focus:
            with torch.no_grad():
                # 计算每个像素的预测置信度
                probs = F.softmax(pred, dim=1)
                max_probs, _ = probs.max(dim=1)
                confidence = 1.0 - max_probs  # 低置信度区域可能是难例

                # 动态调整难例权重
                difficulty_weight = confidence / (confidence.mean() + self.eps)
                difficulty_weight = torch.clamp(difficulty_weight, 0.5, 2.0)

            weighted_ce = (ce_loss * difficulty_weight * adjusted_weights[target]).mean()

        # 添加Dice损失增强边界处理
        dice_loss = self.dice_loss(pred, target)

        # 自动平衡组合
        total_loss = weighted_ce + dice_loss

        return total_loss

    def dice_loss(self, pred, target):
        smooth = 1.0
        pred = F.softmax(pred, dim=1)
        one_hot_target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred * one_hot_target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + one_hot_target.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss