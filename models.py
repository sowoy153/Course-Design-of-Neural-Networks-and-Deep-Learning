import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class WideResNet(nn.Module):
#     def __init__(self, depth=28, widen_factor=10, num_classes=10):
#         super(WideResNet, self).__init__()
#         self.in_planes = 16
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.block1 = self._make_layer(depth, 16 * widen_factor, 1)
#         self.block2 = self._make_layer(depth, 32 * widen_factor, 2)
#         self.block3 = self._make_layer(depth, 64 * widen_factor, 2)
#         self.bn1 = nn.BatchNorm2d(64 * widen_factor)
#         self.fc = nn.Linear(64 * widen_factor, num_classes)

#     def _make_layer(self, depth, planes, stride=1):
#         layers = []
#         for i in range(depth):
#             layers.append(nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(planes))
#             layers.append(nn.ReLU(inplace=True))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = F.relu(self.bn1(out))
#         out = F.avg_pool2d(out, out.size()[2:])  # 使用全局平均池化
#         out = out.view(out.size(0), -1)  # 展平
#         out = self.fc(out)
#         return out

class WideResNet(nn.Module):
    def __init__(self, depth=10, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(depth, 16 * widen_factor, 1)
        self.block2 = self._make_layer(depth, 32 * widen_factor, 2)
        self.block3 = self._make_layer(depth, 64 * widen_factor, 2)
        self.bn1 = nn.BatchNorm2d(64 * widen_factor)
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def _make_layer(self, depth, planes, stride=1):
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2:])  # 使用全局平均池化
        out = out.view(out.size(0), -1)  # 展平
        out = self.fc(out)
        return out