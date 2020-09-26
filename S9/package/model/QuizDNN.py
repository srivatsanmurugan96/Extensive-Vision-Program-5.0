import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # Depthwise seperable convoution
                nn.Conv2d(in_planes, in_planes, kernel_size=(3, 3), padding=1, groups=in_planes),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(self.expansion * planes),
            )
            self.MaxPool = nn.Sequential(
                nn.MaxPool2d(2, 2)
            )
        else:
            self.shortcut = nn.Sequential()
            self.MaxPool = nn.Sequential()

    def forward(self, x):
        ot = self.MaxPool(x)
        out = self.bn1(F.relu(self.conv1(ot)))
        out += self.shortcut(ot)
        out = F.relu(out)

        return out


class QuizNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QuizNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)

        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )  # output_size = 1

        self.linear = nn.Linear(256, 10, bias=False)

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
        out = self.GAP(out)
        out = out.view(-1, 256)
        out = self.linear(out)

        return out


def Quiz_DNN():
    return QuizNet(BasicBlock, [1, 2, 2])