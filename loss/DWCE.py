import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightedCrossEntropyLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(DynamicWeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Compute the dynamic weighted cross entropy loss.

        Args:
            logits: [B, C, H, W] Tensor, raw, unnormalized scores for each class.
            labels: [B, H, W] Tensor, ground truth labels with class indices.

        Returns:
            loss: The dynamic weighted cross entropy loss.
        """
        # Compute the number of pixels for each class
        class_counts = torch.bincount(labels.view(-1), minlength=logits.size(1)).float()
        total_count = labels.numel()

        # Compute class weights: inverse of class frequency
        weights = total_count / (class_counts + 1e-6)
        # Compute weighted cross entropy loss
        loss = F.cross_entropy(logits, labels, weight=weights,reduction=self.reduction)
        return loss


# Example usage
if __name__ == "__main__":
    criterion = DynamicWeightedCrossEntropyLoss()
    logits = torch.randn(8, 2, 256, 256)  # Example logits
    labels = torch.randint(0, 2, (8, 256, 256))  # Example labels
    loss = criterion(logits, labels)
    print(loss)


