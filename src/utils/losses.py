import logging
import torch.nn.functional as F
from typing import List
import torch.nn as nn
import torch


class MultiTaskCrossEntropyLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)
        return [self.loss_fn(i, t) for i, t in zip(input, target)]


class CityScapesLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn_semantic = SemanticSegmentationLoss()
        self.loss_fn_instance = InstanceSegmentationLoss()
        self.loss_fn_depth = DepthLoss()

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input[0], target[0])
        i = self.loss_fn_instance(input[1], target[1])
        d = self.loss_fn_depth(input[2], target[2])
        return [s, i, d]


class UTKFaceMultiTaskLoss(nn.modules.loss._Loss):
    def __init__(self, use_gender_task=True):
        super().__init__()
        self.use_gender_task = use_gender_task
        self.huber = nn.HuberLoss(delta=1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target) -> List[torch.Tensor]:
        loss = []
        if self.use_gender_task:
            # gender task is always first
            loss.append(self.huber(input[0].squeeze(), target[0].squeeze()))
            input = input[1:]
            target = target[1:]
        if isinstance(input, (list, tuple)):
            for i, t in zip(input, target):
                loss.append(self.ce(i, t))
        if isinstance(input, torch.Tensor):
            loss.append(self.ce(input, target))

        return loss


class DepthLoss(nn.Module):
    """Loss for depth prediction. By default L1 loss is used."""

    def __init__(self, loss="l1", ignore_value=0.0):
        super(DepthLoss, self).__init__()
        self.ignore_value = ignore_value
        if loss == "l1":
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError("Loss {} currently not supported in DepthLoss".format(loss))

    def forward(self, out, label):
        if out.shape != label.shape:
            label = label.unsqueeze(1)
        mask = label != self.ignore_value
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class InstanceSegmentationLoss(nn.Module):
    """ """

    def __init__(self, loss="l1", ignore_value=250):
        super(InstanceSegmentationLoss, self).__init__()
        self.ignore_value = ignore_value
        if loss == "l1":
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError("Loss {} currently not supported in InstanceSegmentationLoss".format(loss))

    def forward(self, out, label):
        mask = label != self.ignore_value
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class SemanticSegmentationLoss(nn.CrossEntropyLoss):
    def __init__(self, size_average=True, ignore_index: int = 250, reduction: str = "mean") -> None:
        super().__init__(size_average=size_average, ignore_index=ignore_index, reduction=reduction)


class CityscapesTwoTaskLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def loss_fn_semantic(self, pred, target):
        target = target.long()
        return F.nll_loss(pred, target, ignore_index=-1)

    def loss_fn_depth(self, pred, target):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - target) * binary_mask)
        loss = loss / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input[0], target[0])
        d = self.loss_fn_depth(input[1], target[1])
        return [s, d]


class NYUv2Loss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def loss_fn_semantic(self, pred, target):
        target = target.long()
        return F.nll_loss(pred, target, ignore_index=-1)

    def loss_fn_depth(self, pred, target):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - target) * binary_mask)
        loss = loss / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss

    def loss_fn_normal(self, pred, target):
        binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum((pred * target) * binary_mask)
        loss = loss / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return 1 - loss

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input[0], target[0])
        d = self.loss_fn_depth(input[1], target[1])
        n = self.loss_fn_normal(input[2], target[2])
        return [s, d, n]


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
            0
        )

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss
