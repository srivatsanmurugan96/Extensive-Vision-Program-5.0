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


class CIFAR_Depthwise_Separable_Model(nn.Module):

    def __init__(self):
        super(CIFAR_Depthwise_Separable_Model, self).__init__()
        # Input Block

        self.drop_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        # Conv1 -Input-32x32x3, Output-32x32x32, RF-3
        # Conv2 -Input-32x32x32, Output-32x32x128, RF-5

        self.MP1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False))

        # Max Pooling 1- Input-32x32x128, Output-16x16x128, RF-6
        # Transition Layer 1- Input-16x16x128, Output-16x16x32 , RF-6

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        # Conv3 - Input-16x16x32, Output-16x16x64 , RF-10
        # Conv4 - Input-16x16x64, Output-16x16x256 , RF-14

        self.MP2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), bias=False)
        )
        # Max Pooling 2 - Input- 16x16x256 , Output-8x8x256, RF-16
        # Transition Layer 2 -Input- 8x8x256 , Output-8x8x64, RF-16

        # Convolution Block 3 - Depthwise separable convolution and Dilation
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
        )
        # Conv5 - Input- 8x8x64 , Output-8x8x128, RF-24 ( Depthwise separable convolution )
        # Conv6 - Input- 8x8x128 , Output-8x8x256, RF-40 ( Dilation )

        self.MP3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), bias=False)
        )
        # Max Pooling 3 - Input- 8x8x256 , Output-4x4x256, RF-44
        # Transition Layer 2 - Input- 4x4x256 , Output-4x4x32, RF-44

        # Convolution Block 4 - Depthwise separable convolution
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(3, 3), groups=32, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.drop_val),
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        # Conv7 - Input- 4x4x32 , Output-4x4x256, RF-60 ( Depthwise separable convolution )
        # Global average pooling - Input- 4x4x256 , Output-1x1x256 , RF-84
        # Final - Input- 1x1x256 , Output-1x1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.MP1(x)
        x = self.convblock2(x)
        x = self.MP2(x)
        x = self.convblock3(x)
        x = self.MP3(x)
        x = self.convblock4(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)