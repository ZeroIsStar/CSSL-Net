import torch
from loss.lovasz import lovasz_softmax
from torch import nn


class Lovasz_ce_loss(torch.nn.Module):
    def __init__(self, n_class= 2, weight = None):
        super(Lovasz_ce_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, labels):
        losm_loss = lovasz_softmax(outputs, labels)
        loss = losm_loss
        return loss

    def __call__(self, outputs, labels):
        return self.forward(outputs, labels)


# =======Loss=====#
# Tl = TL_loss(n_class=2).cuda()
# inputs = torch.rand(8,2,256,256)
# target = torch.randint(0,2,(8,256,256))
# a = Tl(inputs, target)
# print(a)

