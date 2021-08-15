import torch.nn as nn
import torch.nn.functional as F

# 496 weights
class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=0)
        self.fc1 = nn.Linear(6 * 2 * 2, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

# 1306 weights
class LeNet3_2(nn.Module):
    def __init__(self):
        super(LeNet3_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=0)
        self.fc1 = nn.Linear(12 * 2 * 2, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model_on_cifar10_170(nn.Module):
    def __init__(self):
        super(Model_on_cifar10_170, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.fc1 = nn.Linear(192 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 80000 weights
class Model_on_cifar10(nn.Module):
    def __init__(self):
        super(Model_on_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6,16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x