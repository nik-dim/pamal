import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel


class LeNet(BaseModel):
    """Simple LeNet for debugging purposes."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        b = x.size(0)
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class MultiLeNetR(nn.Module):
    """The encoder part of the LeNet network adapted to MultiTask Learning. The model consists of two convolutions
    followed by a fully connected layers, resulting in a 50-dimensional embedding."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 50)

    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x

    def get_last_layer(self):
        return self.fc


class MultiLeNetO(nn.Module):
    """The decoder part of the LeNet network adapted to MultiTask Learning. The output has 10 dimensions, since this
    model is used for datasets such as MultiMNIST."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
