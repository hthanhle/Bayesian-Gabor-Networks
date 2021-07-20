"""
Created on Mon Aug 17 12:19:56 2020

@author: tlh857
"""

import torch.nn as nn
from layers.BayesianGabor_v3 import BayesianGabor2D


# Note that this version declares a forward function explicitly, where the KL loss is
# accumulated during the forward process. Hence, we do not need to inherit the class
# 'ModuleWrapper' in 'misc.py' 

# This is exactly BayesianGaborPFPN_config3
print('BayesianGaborNetwork-v1')


class BayesianGaborNetwork(nn.Module):
    def __init__(self, num_inputs, num_classes, priors, activation_type='relu'):
        super(BayesianGaborNetwork, self).__init__()

        self.num_classes = num_classes
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.gabor1 = BayesianGabor2D(num_inputs, 32, 15, padding=7, priors=self.priors)
        self.act1 = self.act()
        self.down0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor2 = BayesianGabor2D(32, 32, 15, padding=7, priors=self.priors)
        self.act2 = self.act()
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor3 = BayesianGabor2D(32, 32, 15, padding=7, priors=self.priors)
        self.act3 = self.act()

        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor4 = BayesianGabor2D(32, 64, 7, padding=3, priors=self.priors)
        self.act4 = self.act()
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor5 = BayesianGabor2D(64, 64, 7, padding=3, priors=self.priors)
        self.act5 = self.act()

        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor6 = BayesianGabor2D(64, 64, 5, padding=2, priors=self.priors)
        self.act6 = self.act()
        self.down5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gabor7 = BayesianGabor2D(64, 64, 5, padding=2, priors=self.priors)
        self.act7 = self.act()
        self.up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.gabor8 = BayesianGabor2D(64, 128, 3, padding=1, priors=self.priors)
        self.act8 = self.act()

        self.gabor9 = BayesianGabor2D(64, 64, 5, padding=2, priors=self.priors)
        self.act9 = self.act()
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.gabor10 = BayesianGabor2D(64, 128, 3, padding=1, priors=self.priors)
        self.act10 = self.act()

        self.gabor11 = BayesianGabor2D(32, 64, 5, padding=2, priors=self.priors)
        self.act11 = self.act()
        self.gabor12 = BayesianGabor2D(64, 128, 3, padding=1, priors=self.priors)
        self.act12 = self.act()

        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.gabor13 = BayesianGabor2D(128, 32, 3, padding=1, priors=self.priors)
        self.act13 = self.act()
        self.gabor14 = BayesianGabor2D(32, self.num_classes, 1, padding=0, priors=self.priors)

    def forward(self, input):
        kl = 0.0

        x1 = self.act1(self.gabor1(input))
        kl = kl + self.gabor1.kl_loss()
        x1_down = self.down0(x1)
        x2 = self.act2(self.gabor2(x1_down))
        kl = kl + self.gabor2.kl_loss()
        x2_down = self.down1(x2)
        x3 = self.act3(self.gabor3(x2_down))
        kl = kl + self.gabor3.kl_loss()

        x3_down = self.down2(x3)
        x4 = self.act4(self.gabor4(x3_down))
        kl = kl + self.gabor4.kl_loss()
        x4_down = self.down3(x4)
        x5 = self.act5(self.gabor5(x4_down))
        kl = kl + self.gabor5.kl_loss()

        x5_down = self.down4(x5)
        x6 = self.act6(self.gabor6(x5_down))
        kl = kl + self.gabor6.kl_loss()
        x6_down = self.down5(x6)
        x7 = self.act7(self.gabor7(x6_down))
        kl = kl + self.gabor7.kl_loss()
        x7_up = self.up1(x7)
        x8 = self.act8(self.gabor8(x7_up))
        kl = kl + self.gabor8.kl_loss()

        x9 = self.act9(self.gabor9(x5))
        kl = kl + self.gabor9.kl_loss()
        x9_up = self.up2(x9)
        x10 = self.act10(self.gabor10(x9_up))
        kl = kl + self.gabor10.kl_loss()

        x11 = self.act11(self.gabor11(x3))
        kl = kl + self.gabor11.kl_loss()
        x12 = self.act12(self.gabor12(x11))
        kl = kl + self.gabor12.kl_loss()

        x8 = nn.functional.interpolate(x8, x10.shape[2:], mode='bilinear', align_corners=False)
        x_8_10 = x8 + x10
        x_8_10_12 = x_8_10 + x12

        x_8_10_12_up = self.up3(x_8_10_12)
        x13 = self.act13(self.gabor13(x_8_10_12_up))
        kl = kl + self.gabor13.kl_loss()
        x14 = self.gabor14(x13)
        kl = kl + self.gabor14.kl_loss()

        return x14, kl  # Return a tuple of (logits, kl)
