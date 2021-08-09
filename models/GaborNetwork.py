"""
Created on Wed June 30 12:19:56 2021
(Non-Bayesian) Gabor Network
@author: Thanh Le
"""

import torch.nn as nn
from layers.Gabor_v2 import Gabor2D


class GaborNetwork(nn.Module):
    def __init__(self, num_inputs, num_classes, activation='relu'):
        """
        :param num_inputs: number of input channels
        :param num_classes: number of classes
        :param activation: activation type
        """
        super(GaborNetwork, self).__init__()
        self.num_classes = num_classes
        if activation == 'softplus':
            self.act = nn.Softplus
        elif activation == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or ReLU supported")

        self.gabor1 = Gabor2D(num_inputs, 32, 15, padding=7)
        self.act1 = self.act()
        self.down0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor2 = Gabor2D(32, 32, 15, padding=7)
        self.act2 = self.act()
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor3 = Gabor2D(32, 32, 15, padding=7)
        self.act3 = self.act()

        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor4 = Gabor2D(32, 64, 7, padding=3)
        self.act4 = self.act()
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor5 = Gabor2D(64, 64, 7, padding=3)
        self.act5 = self.act()

        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor6 = Gabor2D(64, 64, 5, padding=2)
        self.act6 = self.act()
        self.down5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor7 = Gabor2D(64, 64, 5, padding=2)
        self.act7 = self.act()
        self.up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.gabor8 = Gabor2D(64, 128, 3, padding=1)
        self.act8 = self.act()

        self.gabor9 = Gabor2D(64, 64, 5, padding=2)
        self.act9 = self.act()
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.gabor10 = Gabor2D(64, 128, 3, padding=1)
        self.act10 = self.act()

        self.gabor11 = Gabor2D(32, 64, 5, padding=2)
        self.act11 = self.act()
        self.gabor12 = Gabor2D(64, 128, 3, padding=1)
        self.act12 = self.act()

        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.gabor13 = Gabor2D(128, 32, 3, padding=1)
        self.act13 = self.act()
        self.gabor14 = Gabor2D(32, self.num_classes, 1, padding=0)

    def forward(self, input):
        x1 = self.act1(self.gabor1(input))
        x1_down = self.down0(x1)
        x2 = self.act2(self.gabor2(x1_down))
        x2_down = self.down1(x2)
        x3 = self.act3(self.gabor3(x2_down))

        x3_down = self.down2(x3)
        x4 = self.act4(self.gabor4(x3_down))
        x4_down = self.down3(x4)
        x5 = self.act5(self.gabor5(x4_down))

        x5_down = self.down4(x5)
        x6 = self.act6(self.gabor6(x5_down))
        x6_down = self.down5(x6)
        x7 = self.act7(self.gabor7(x6_down))
        x7_up = self.up1(x7)
        x8 = self.act8(self.gabor8(x7_up))

        x9 = self.act9(self.gabor9(x5))
        x9_up = self.up2(x9)
        x10 = self.act10(self.gabor10(x9_up))

        x11 = self.act11(self.gabor11(x3))
        x12 = self.act12(self.gabor12(x11))

        x8 = nn.functional.interpolate(x8, x10.shape[2:], mode='bilinear', align_corners=False)
        x_8_10 = x8 + x10
        x_8_10_12 = x_8_10 + x12

        x_8_10_12_up = self.up3(x_8_10_12)
        x13 = self.act13(self.gabor13(x_8_10_12_up))
        x14 = self.gabor14(x13)

        return x14
