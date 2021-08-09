"""
Created on Thu Jun 25 10:17:42 2020
Non-Bayesian Gabor layer
Reference: H. T. Le, S. L. Phung, P. B. Chapple, A. Bouzerdoum, C. H. Ritz, and L. C. Tran, “Deep Gabor neural network for automatic detection of mine-like objects in sonar imagery,” IEEE Access, vol. 8, pp. 94 126–94 139, 2020.
@author: Thanh Le
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class Gabor2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :param dilation: dilation factor
        :param bias: bias
        """
        super(Gabor2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.theta = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.lamda = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.gamma = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.psi = Parameter(torch.empty((out_channels, in_channels), device=self.device))

        if self.use_bias:
            self.bias = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.sigma.data.normal_(5, 1.5)
        self.theta.data.uniform_(0, 1)
        self.lamda.data.normal_(5, 1.5)
        self.gamma.data.normal_(1.5, 0.4)
        self.psi.data.uniform_(0, 1)
        if self.use_bias:
            self.bias.data.normal_(0, 1)

    def forward(self, input, sample=True):
        # Gabor computations
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        half_size = np.floor(self.kernel_size / 2)
        y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
        x_kernels = np.tile(x, (self.out_channels, self.in_channels, 1, 1))
        y_kernels = np.tile(y, (self.out_channels, self.in_channels, 1, 1))

        rot_x_kernels = torch.from_numpy(x_kernels).float().to(self.device) * torch.cos(self.theta - np.pi).unsqueeze(
            2).unsqueeze(2) + torch.from_numpy(y_kernels).float().to(self.device) * torch.sin(
            self.theta - np.pi).unsqueeze(2).unsqueeze(2)
        rot_y_kernels = torch.from_numpy(-x_kernels).float().to(self.device) * torch.sin(self.theta - np.pi).unsqueeze(
            2).unsqueeze(2) + torch.from_numpy(y_kernels).float().to(self.device) * torch.cos(
            self.theta - np.pi).unsqueeze(2).unsqueeze(2)

        # Compute the real parts of Gabor kernels
        gabor_kernels = torch.exp(-0.5 * (rot_x_kernels ** 2 / sigma_x.unsqueeze(2).unsqueeze(
            2) ** 2 + rot_y_kernels ** 2 / sigma_y.unsqueeze(2).unsqueeze(2) ** 2))
        gabor_kernels = gabor_kernels / (
                    2 * np.pi * sigma_x.unsqueeze(2).unsqueeze(2) * sigma_y.unsqueeze(2).unsqueeze(2))
        gabor_kernels = gabor_kernels * torch.cos(
            2 * np.pi / self.lamda.unsqueeze(2).unsqueeze(2) * rot_x_kernels + self.psi.unsqueeze(2).unsqueeze(2))

        return F.conv2d(input, gabor_kernels, self.bias, self.stride, self.padding, self.dilation, self.groups)
