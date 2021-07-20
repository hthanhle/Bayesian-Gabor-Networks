"""
Created on Thu Jun 25 10:17:42 2020
Standard Gabor layer (same as the Keras 'gabor_layer_v4')
Version 2: Optimize the computations to avoid overloading CPUs
@author: Thanh Le
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

print('Using Gabor-v2 layer')

class Gabor2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):

        super(Gabor2D, self).__init__()
        self.in_channels = in_channels  # number of input channels
        self.out_channels = out_channels  # number of kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the trainable weights. Note that a weight denotes the distribution for each Gabor parameter
        self.sigma = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.theta = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.lamda = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.gamma = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.psi = Parameter(torch.empty((out_channels, in_channels), device=self.device))

        # Create the 4D mesh grid of size [out_channels, in_channels, kernel_size, kernel_size]
        half_size = np.floor(self.kernel_size / 2)
        y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
        x_kernels = np.tile(x, (out_channels, in_channels, 1, 1))  # 4D Numpy Array
        y_kernels = np.tile(y, (out_channels, in_channels, 1, 1))
        self.x_kernels = torch.from_numpy(x_kernels).float().to(
            self.device)  # hereafter, they are tensors on GPU (no computations performed on CPU)
        self.y_kernels = torch.from_numpy(y_kernels).float().to(self.device)

        # Use bias
        if self.use_bias:
            self.bias = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    # In[2]: Initialize the parameters
    def reset_parameters(self):
        self.sigma.data.normal_(5, 1.5)
        self.theta.data.uniform_(0, 1)
        self.lamda.data.normal_(5, 1.5)
        self.gamma.data.normal_(1.5, 0.4)
        self.psi.data.uniform_(0, 1)
        if self.use_bias:
            self.bias.data.normal_(0, 1)

    # In[3]: Define actual computations
    def forward(self, input, sample=True):
        # Gabor computations
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        # The variables 'rot_x_kernels' and 'rot_y_kernels' are 4D tensors. Note that '.unsqueeze(2).unsqueeze(2)' is to expand the tensors along two last dimensions. For example: [5, 3] --> [5, 3, 1, 1]. 
        # Looking back the Gabor-v3, Tensorflow does not need to convert 'x_kernels' to a Numpy array, and to expand the dimensions.
        rot_x_kernels = self.x_kernels * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(
            2) + self.y_kernels * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(2)
        rot_y_kernels = -self.x_kernels * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(
            2) + self.y_kernels * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(2)

        # Compute the real parts of Gabor kernels as OpenCV-style
        gabor_kernels = torch.exp(-0.5 * (rot_x_kernels ** 2 / sigma_x.unsqueeze(2).unsqueeze(
            2) ** 2 + rot_y_kernels ** 2 / sigma_y.unsqueeze(2).unsqueeze(2) ** 2))
        gabor_kernels = gabor_kernels / (2 * np.pi * sigma_x.unsqueeze(2).unsqueeze(2) * sigma_y.unsqueeze(2).unsqueeze(
            2))  # Must not use inplace operations like '/=' or '*='
        gabor_kernels = gabor_kernels * torch.cos(
            2 * np.pi / self.lamda.unsqueeze(2).unsqueeze(2) * rot_x_kernels + self.psi.unsqueeze(2).unsqueeze(2))

        return F.conv2d(input, gabor_kernels, self.bias, self.stride, self.padding, self.dilation, self.groups)
