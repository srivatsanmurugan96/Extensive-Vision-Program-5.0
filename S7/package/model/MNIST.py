from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        # Input Block 1
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # ouput - 26

        self.BatchN1 = nn.BatchNorm2d(10)

        self.GBN1 = GhostBatchNorm(10, 2)

        # Convolution Block 1
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # ouput - 24

        self.BatchN2 = nn.BatchNorm2d(16)

        self.GBN2 = GhostBatchNorm(16, 2)

        # Transistion Block 1
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # ouput - 12

        self.BatchN3 = nn.BatchNorm2d(10)

        self.GBN3 = GhostBatchNorm(10, 2)

        # Convolution Block 2

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # ouput - 10

        self.BatchN4 = nn.BatchNorm2d(16)

        self.GBN4 = GhostBatchNorm(16, 2)

        # Convolution Block 3

        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()

        )  # ouput - 8

        self.BatchN5 = nn.BatchNorm2d(16)

        self.GBN5 = GhostBatchNorm(16, 2)

        # Convolution Block 4

        self.Conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # ouput - 6

        self.BatchN6 = nn.BatchNorm2d(16)

        self.GBN6 = GhostBatchNorm(16, 2)

        # Convolution Block 5

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # ouput - 1

        # Convolution Block 6

        self.Conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x, Batch_Norm):
        x = self.Conv1(x)
        if Batch_Norm == 'BN':
            x = self.BatchN1(x)
        else:
            x = self.GBN1(x)
        x = self.Conv2(x)
        if Batch_Norm == 'BN':
            x = self.BatchN2(x)
        else:
            x = self.GBN2(x)
        x = self.pool1(x)
        x = self.Conv3(x)
        if Batch_Norm == 'BN':
            x = self.BatchN3(x)
        else:
            x = self.GBN3(x)
        x = self.Conv4(x)
        if Batch_Norm == 'BN':
            x = self.BatchN4(x)
        else:
            x = self.GBN4(x)
        x = self.Conv5(x)
        if Batch_Norm == 'BN':
            x = self.BatchN5(x)
        else:
            x = self.GBN5(x)
        x = self.Conv6(x)
        if Batch_Norm == 'BN':
            x = self.BatchN6(x)
        else:
            x = self.GBN6(x)
        x = self.GAP(x)
        x = self.Conv7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)