import torch

def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou_score = intersection / union
    return iou_score.item()

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice_score = (2. * intersection) / (pred.sum() + target.sum())
    return dice_score.item()

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    correct = (pred == target).sum()
    accuracy_score = correct / torch.numel(pred)
    return accuracy_score.item()
