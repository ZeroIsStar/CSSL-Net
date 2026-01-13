from loss.DWCE import DynamicWeightedCrossEntropyLoss
from loss.focal import FocalLoss
from loss.dice import DiceLoss
from loss.lovasz import LovaszSoftmaxLoss
import torch

class mix_loss(torch.nn.Module):
    def __init__(self):
        super(mix_loss, self).__init__()
        self.dwce = DynamicWeightedCrossEntropyLoss()
        self.focal = FocalLoss()
        self.dice = DiceLoss(mode="multiclass")

    def forward(self, outputs, labels, epoch):
        dice_loss = self.dice(outputs, labels)
        lovaszSoftmax = LovaszSoftmaxLoss(outputs, labels)
        loss = 0.5*lovaszSoftmax + 0.5*dice_loss
        return loss

    def __call__(self, outputs, labels, epoch):
        return self.forward(outputs, labels, epoch)


# Example usage
if __name__ == "__main__":
    criterion = mix_loss()
    logits = torch.randn(8, 2, 256, 256)  # Example logits
    labels = torch.randint(0, 2, (8, 256, 256))  # Example labels
    loss = criterion(logits, labels)
    print(loss)