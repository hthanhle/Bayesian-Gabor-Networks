"""
Created on Mon Aug 17 12:19:56 2020

@author: tlh857
"""

import torch.nn as nn
from layers.BayesianConv2D import BayesianConv2D

# A convolutional Bayesian CNN counterpart of BayesianGaborPFPN_config3. This aims to evaluate the model with KL, without conv
print('Bayesian CNN with None prior')


class BayesianCNN(nn.Module):
    def __init__(self, num_inputs, num_classes, priors, activation_type='relu'):
        super(BayesianCNN, self).__init__()

        self.num_classes = num_classes
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BayesianConv2D(num_inputs, 32, 15, padding=7, priors=None)
        self.act1 = self.act()
        self.down0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BayesianConv2D(32, 32, 15, padding=7, priors=None)
        self.act2 = self.act()
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = BayesianConv2D(32, 32, 15, padding=7, priors=None)
        self.act3 = self.act()

        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = BayesianConv2D(32, 64, 7, padding=3, priors=None)
        self.act4 = self.act()
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = BayesianConv2D(64, 64, 7, padding=3, priors=None)
        self.act5 = self.act()

        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = BayesianConv2D(64, 64, 5, padding=2, priors=None)
        self.act6 = self.act()
        self.down5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = BayesianConv2D(64, 64, 5, padding=2, priors=None)
        self.act7 = self.act()
        self.up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.conv8 = BayesianConv2D(64, 128, 3, padding=1, priors=None)
        self.act8 = self.act()

        self.conv9 = BayesianConv2D(64, 64, 5, padding=2, priors=None)
        self.act9 = self.act()
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv10 = BayesianConv2D(64, 128, 3, padding=1, priors=None)
        self.act10 = self.act()

        self.conv11 = BayesianConv2D(32, 64, 5, padding=2, priors=None)
        self.act11 = self.act()
        self.conv12 = BayesianConv2D(64, 128, 3, padding=1, priors=None)
        self.act12 = self.act()

        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv13 = BayesianConv2D(128, 32, 3, padding=1, priors=None)
        self.act13 = self.act()
        self.conv14 = BayesianConv2D(32, self.num_classes, 1, padding=0, priors=None)

    def forward(self, input):
        kl = 0.0

        x1 = self.act1(self.conv1(input))
        kl = kl + self.conv1.kl_loss()
        x1_down = self.down0(x1)
        x2 = self.act2(self.conv2(x1_down))
        kl = kl + self.conv2.kl_loss()
        x2_down = self.down1(x2)
        x3 = self.act3(self.conv3(x2_down))
        kl = kl + self.conv3.kl_loss()

        x3_down = self.down2(x3)
        x4 = self.act4(self.conv4(x3_down))
        kl = kl + self.conv4.kl_loss()
        x4_down = self.down3(x4)
        x5 = self.act5(self.conv5(x4_down))
        kl = kl + self.conv5.kl_loss()

        x5_down = self.down4(x5)
        x6 = self.act6(self.conv6(x5_down))
        kl = kl + self.conv6.kl_loss()
        x6_down = self.down5(x6)
        x7 = self.act7(self.conv7(x6_down))
        kl = kl + self.conv7.kl_loss()
        x7_up = self.up1(x7)
        x8 = self.act8(self.conv8(x7_up))
        kl = kl + self.conv8.kl_loss()

        x9 = self.act9(self.conv9(x5))
        kl = kl + self.conv9.kl_loss()
        x9_up = self.up2(x9)
        x10 = self.act10(self.conv10(x9_up))
        kl = kl + self.conv10.kl_loss()

        x11 = self.act11(self.conv11(x3))
        kl = kl + self.conv11.kl_loss()
        x12 = self.act12(self.conv12(x11))
        kl = kl + self.conv12.kl_loss()

        x8 = nn.functional.interpolate(x8, x10.shape[2:], mode='bilinear', align_corners=False)
        x_8_10 = x8 + x10
        x_8_10_12 = x_8_10 + x12

        x_8_10_12_up = self.up3(x_8_10_12)
        x13 = self.act13(self.conv13(x_8_10_12_up))
        kl = kl + self.conv13.kl_loss()
        x14 = self.conv14(x13)
        kl = kl + self.conv14.kl_loss()

        return x14, kl  # Return a tuple of (logits, kl)
