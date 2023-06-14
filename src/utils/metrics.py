import numpy as np

import torchmetrics
from torch import Tensor
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex


class MaskedL1Metric(torchmetrics.MeanAbsoluteError):
    def __init__(self, ignore_index, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            target = target.unsqueeze(1)
        mask = target != self.ignore_index
        return super().update(torch.masked_select(preds, mask), torch.masked_select(target, mask))


class CrossEntropyLossMetric(torchmetrics.MeanMetric):
    def __init__(self, ignore_index=-100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def update(self, preds, target) -> None:
        value = F.cross_entropy(input=preds, target=target, ignore_index=self.ignore_index)
        return super().update(value)


class HuberLossMetric(torchmetrics.MeanMetric):
    def update(self, preds, target) -> None:
        value = F.huber_loss(input=preds.squeeze(), target=target, delta=1)
        return super().update(value)


class DummyMetric(torchmetrics.MaxMetric):
    def update(self, value, *args, **kwargs) -> None:
        return super().update(123)


class ModifiedJaccardIndex(JaccardIndex):
    def __init__(self):
        super().__init__(compute_on_step=False, num_classes=19, ignore_index=250)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        mask = target != self.ignore_index
        preds = preds.argmax(1)
        return super().update(torch.masked_select(preds, mask), torch.masked_select(target, mask))
