import torch
import torch.nn as nn
from loss.lovasz import LovaszSoftmaxLoss
from loss.Tversky_loss import Tversky_loss


class AutoBalanceWeightedLoss(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.tversky_loss = Tversky_loss(clsasses=classes)

    def forward(self, preds, targets):
        pred_prob = torch.sigmoid(preds)
        confidence = torch.abs(pred_prob - 0.5)  # 离决策边界越近越困难
        hardness = 1.0 - confidence
        # 计算当前损失值
        loss_tversky = self.tversky_loss(preds, targets)
        loss_lovasz = LovaszSoftmaxLoss(preds, targets)
        total_loss = ((0.5*loss_tversky + 0.5*loss_lovasz)*hardness).mean()
        return total_loss
