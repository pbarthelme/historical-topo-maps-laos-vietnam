import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicCrossEntropyLoss(nn.Module):
    """Wrapper function around torch cross-entropy loss manually applying mean reduction step."""
    def __init__(self, weight=None, ignore_index=None):
        super(DeterministicCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Compute per-sample loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight, ignore_index=self.ignore_index)

        return ce_loss.mean()
    
class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation allowing per class alpha values."""
    def __init__(self, alpha=None, gamma=2, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_part = (1 - pt) ** self.gamma
        loss = (self.alpha[targets] * focal_part * ce_loss).mean()
        return loss
    
