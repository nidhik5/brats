"""Dice + categorical focal loss (PyTorch equivalent of the notebook's
segmentation_models_3D DiceLoss(class_weights=[.25]*4) + CategoricalFocalLoss)."""
import torch
import torch.nn.functional as F


def soft_dice_loss(probs, onehot, class_weights, eps=1e-5):
    # Dice per class, pooled over batch and space (sm's per_image=False).
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - (dice * class_weights).sum() / class_weights.sum()


def focal_loss(probs, onehot, gamma=2.0, alpha=0.25, eps=1e-7):
    p = probs.clamp(eps, 1 - eps)
    loss = -onehot * alpha * (1 - p) ** gamma * torch.log(p)
    return loss.sum(1).mean()


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, num_classes=4, class_weights=None, focal_weight=1.0):
        super().__init__()
        w = torch.ones(num_classes) if class_weights is None else torch.as_tensor(class_weights)
        self.register_buffer("class_weights", w.float())
        self.num_classes = num_classes
        self.focal_weight = focal_weight

    def forward(self, logits, target):
        # Loss in float32 regardless of autocast.
        probs = F.softmax(logits.float(), dim=1)
        onehot = F.one_hot(target.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()
        return (soft_dice_loss(probs, onehot, self.class_weights)
                + self.focal_weight * focal_loss(probs, onehot))
