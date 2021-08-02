import torch
import torch.nn as nn
import torch.nn.functional as F

EXPANSION = 4


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, EXPANSION * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(EXPANSION * planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(EXPANSION * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        woha = self.shortcut(x)
        out += woha
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)  # 3 times

        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)  # 4 times

        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)  # 6 times

        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)  # 3 times

        self.linear = nn.Conv2d(EXPANSION * 512, num_classes, kernel_size=1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = EXPANSION * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = self.linear(out)
        return out


def ResNet50(num_classes):
    return ResNet([3, 4, 6, 3], num_classes)
