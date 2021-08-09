"""
Created on Thu Feb 04 10:17:42 2021
Version 3: Update the prior with the mixture of two Gaussian densities
@author: Thanh Le
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils.metrics import calculate_kl_loss
import numpy as np


class BayesianGabor2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, priors=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :param dilation: dilation factor
        :param bias: bias
        :param priors: priors
        """
        super(BayesianGabor2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prior_sigma1 = priors['prior_sigma1']
        self.prior_sigma2 = priors['prior_sigma2']
        self.psi_mu = Parameter(torch.empty((out_channels, in_channels)))
        self.psi_rho = Parameter(torch.empty((out_channels, in_channels)))

        self.theta = Parameter(torch.empty((out_channels, in_channels)))
        self.lamda = Parameter(torch.empty((out_channels, in_channels)))
        self.sigma = Parameter(torch.empty((out_channels, in_channels)))
        self.gamma = Parameter(torch.empty((out_channels, in_channels)))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels)))
            self.bias_rho = Parameter(torch.empty((out_channels)))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

        half_size = np.floor(self.kernel_size / 2)
        y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
        x_kernels = np.tile(x, (self.out_channels, self.in_channels, 1, 1))
        y_kernels = np.tile(y, (self.out_channels, self.in_channels, 1, 1))
        self.x_kernels = torch.from_numpy(x_kernels).float().to(self.device)
        self.y_kernels = torch.from_numpy(y_kernels).float().to(self.device)

        self.eps = torch.empty(self.psi_mu.size(), device=self.device)

        if self.use_bias:
            self.bias_eps = torch.empty(self.bias_mu.size(), device=self.device)
        else:
            self.bias = None

    def reset_parameters(self):
        self.psi_mu.data.normal_(0, 1)
        self.psi_rho.data.normal_(0, 1)

        self.theta.data.uniform_(0, 1)
        self.sigma.data.normal_(5, 1.5)
        self.lamda.data.normal_(5, 1.5)
        self.gamma.data.normal_(1.5, 0.4)

        if self.use_bias:
            self.bias_mu.data.normal_(0, 1)
            self.bias_rho.data.normal_(0, 1)

    def forward(self, input, sample=True):

        # Reparameterization trick
        eps = self.eps.normal_(0, 1)

        # Draw random samples from the variational porterior distributions
        self.psi_sigma = torch.log1p(torch.exp(self.psi_rho))
        self.psi = (
                self.psi_mu + eps * self.psi_sigma).float()

        if self.use_bias:
            bias_eps = self.bias_eps.normal_(0, 1)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            self.bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            self.bias = None

        # Gabor computations
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        rot_x_kernels = self.x_kernels * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(
            2) + self.y_kernels * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(2)
        rot_y_kernels = -self.x_kernels * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(
            2) + self.y_kernels * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(2)

        # Compute the real parts of Gabor kernels
        gabor_kernels = torch.exp(-0.5 * (rot_x_kernels ** 2 / sigma_x.unsqueeze(2).unsqueeze(
            2) ** 2 + rot_y_kernels ** 2 / sigma_y.unsqueeze(2).unsqueeze(2) ** 2))
        gabor_kernels = gabor_kernels / (
                    2 * np.pi * sigma_x.unsqueeze(2).unsqueeze(2) * sigma_y.unsqueeze(2).unsqueeze(2))
        gabor_kernels = gabor_kernels * torch.cos(
            2 * np.pi / self.lamda.unsqueeze(2).unsqueeze(2) * rot_x_kernels + self.psi.unsqueeze(2).unsqueeze(2))

        return F.conv2d(input, gabor_kernels, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = calculate_kl_loss(self.psi, self.psi_mu, self.psi_sigma, self.prior_sigma1, self.prior_sigma2)

        if self.use_bias:
            kl += calculate_kl_loss(self.bias, self.bias_mu, self.bias_sigma, self.prior_sigma1, self.prior_sigma2)
        return kl
