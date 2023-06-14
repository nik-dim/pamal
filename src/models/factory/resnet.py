"""from Senser and Koltun git repo
"""

import logging
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import SharedBottom


class BasicBlock(nn.Module):
    """BasicBlock block for the Resnet. Adapted from official Pytorch source code."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for the Resnet. Adapted from official Pytorch source code."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet architecture adapted from official Pytorch source code. The main difference lies in the omission of the last layer (the fully connected one), since this resnet is meant for creating an embedding for MultiTask Learning. Used in [1].

    References:
        [1] O. Sener and V. Koltun, “Multi-Task Learning as Multi-Objective Optimization,” in NeurIPS 2018.
    """

    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def get_last_layer(self):
        return self.layer4


def resnet18():
    import torchvision

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Identity()
    return model


class FaceAttributeDecoder(nn.Module):
    """The decoder part succeedind the ResNetEncoder. Used in [1] for the Celeb-A dataset.

    References:
        [1] O. Sener and V. Koltun, “Multi-Task Learning as Multi-Objective Optimization,” in NeurIPS 2018.
    """

    def __init__(self, in_features=2048):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x):
        x = self.linear(x)
        # x = F.log_softmax(x, dim=1)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


class CelebAResnet(SharedBottom):
    def __init__(self, num_tasks=40):
        encoder = ResNetEncoder(block=BasicBlock, num_blocks=[2, 2, 2, 2])
        decoder = FaceAttributeDecoder()
        super().__init__(encoder=encoder, decoder=decoder, num_tasks=num_tasks)


class UtkFaceResnet(SharedBottom):
    def __init__(self, in_channels=3):
        encoder = ResNetEncoder(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_channels=in_channels)
        decoder = [MLPDecoder(num_classes=1), MLPDecoder(num_classes=2), MLPDecoder(num_classes=5)]
        super().__init__(encoder=encoder, decoder=decoder, num_tasks=3)
