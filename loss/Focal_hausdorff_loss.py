import torch
from loss.focal import FocalLoss
from loss.HausdorffERLoss import HausdorffDTLoss


class focal_hausdorffErloss(torch.nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2):
        super(focal_hausdorffErloss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma= gamma)
        self.HausdorffDTloss = HausdorffDTLoss()

    def forward(self, outputs, labels):
        focal_loss = self.focal_loss(outputs, labels)
        hausdorff_loss = self.HausdorffDTloss(outputs, labels)
        loss = 1*focal_loss + 0*hausdorff_loss  # 两者加起来得到最终的loss
        return loss


# label = torch.rand(8, 2, 256, 256).cuda()  # 256x256大小的标签
# pre = torch.randint(0, 2, (8, 256, 256), dtype=torch.long).cuda() # 256x256大小的标签
# loss = focal_hausdorffErloss().to('cuda')
# print(loss(label, pre))
