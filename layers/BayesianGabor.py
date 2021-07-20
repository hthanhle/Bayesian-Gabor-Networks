"""
Created on Thu Jun 25 10:17:42 2020

@author: tlh857
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils.metrics import calculate_kl as KL_DIV
import numpy as np

#print('Only the parameter Psi Bayesianized')
class BayesianGabor2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, priors=None):

        super(BayesianGabor2D, self).__init__()
        self.in_channels = in_channels   # number of input channels
        self.out_channels = out_channels # number of kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        # Define the trainable weights. Note that a weight denotes the distribution for each Gabor parameter
        self.psi_mu  = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        self.psi_rho = Parameter(torch.empty((out_channels, in_channels), device=self.device))
                
        self.theta  = Parameter(torch.empty((out_channels, in_channels), device=self.device))       
        self.lamda  = Parameter(torch.empty((out_channels, in_channels), device=self.device))   
        self.sigma  = Parameter(torch.empty((out_channels, in_channels), device=self.device))    
        self.gamma  = Parameter(torch.empty((out_channels, in_channels), device=self.device))
        
        
        if self.use_bias:
            self.bias_mu  = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # Initialize the parameters
        self.reset_parameters()
        
    # In[2]: Initialize the parameters
    def reset_parameters(self):
        self.psi_mu.data.normal_(*self.posterior_mu_initial) # The symbol '*' denotes a tuple (?)
        self.psi_rho.data.normal_(*self.posterior_rho_initial)
        
        self.theta.data.uniform_(0, 1)
        self.sigma.data.normal_(5, 1.5)      
        self.lamda.data.normal_(5, 1.5)      
        self.gamma.data.normal_(1.5, 0.4)      
                
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)
            
    # In[3]: Define actual computations
    def forward(self, input, sample=True):
        
        # Reparameterization trick: Draw a sample from the standard normal distribution (weight = mu + eps * sigma)
        eps = torch.empty(self.psi_mu.size()).normal_(0, 1).to(self.device) # Note that 'sigma_mu' and other 'mu' have the same size, so we only need to create a common 'eps'        
        # Draw random samples from the variational porterior distributions (i.e. weight = mu + eps * sigma)
        self.psi_sigma = torch.log1p(torch.exp(self.psi_rho)) # Sofplus function. Note that 'sigma_sigma' denotes the deviation of the variational distribution of the parameter 'sigma'
        self.psi = (self.psi_mu + eps * self.psi_sigma).float() # 'sigma' is a sample drawn from the distribution. Its shape is identical to the shape of 'sigma_mu' and 'sigma_rho'.
                       
        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            self.bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            self.bias = None
            
        # Gabor computations
        sigma_x = self.sigma
        sigma_y = self.sigma/self.gamma
        
        # Create the 4D mesh grid of size [out_channels, in_channels, kernel_size, kernel_size]    
        half_size = np.floor(self.kernel_size/2)
        y,x       = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]	
        x_kernels = np.tile(x, (self.out_channels, self.in_channels, 1, 1)) # 4D Numpy Array
        y_kernels = np.tile(y, (self.out_channels, self.in_channels, 1, 1))
        
        # The variables 'rot_x_kernels' and 'rot_y_kernels' are 4D tensors. Note that '.unsqueeze(2).unsqueeze(2)' is to expand the tensors along two last dimensions. For example: [5, 3] --> [5, 3, 1, 1]. 
        # Looking back the Gabor-v3, Tensorflow does not need to convert 'x_kernels' to a Numpy array, and to expand the dimensions.
        rot_x_kernels = torch.from_numpy(x_kernels).float().to(self.device)  * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(2) + torch.from_numpy(y_kernels).float().to(self.device) * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(2)
        rot_y_kernels = torch.from_numpy(-x_kernels).float().to(self.device) * torch.sin(self.theta - np.pi).unsqueeze(2).unsqueeze(2) + torch.from_numpy(y_kernels).float().to(self.device) * torch.cos(self.theta - np.pi).unsqueeze(2).unsqueeze(2)
        
        # Compute the real parts of Gabor kernels as OpenCV-style
        gabor_kernels = torch.exp(-0.5 * (rot_x_kernels ** 2 / sigma_x.unsqueeze(2).unsqueeze(2) ** 2 + rot_y_kernels ** 2 / sigma_y.unsqueeze(2).unsqueeze(2) ** 2))
        gabor_kernels = gabor_kernels/ (2 * np.pi * sigma_x.unsqueeze(2).unsqueeze(2) * sigma_y.unsqueeze(2).unsqueeze(2)) # Must not use inplace operations like '/=' or '*='
        gabor_kernels = gabor_kernels * torch.cos(2 * np.pi / self.lamda.unsqueeze(2).unsqueeze(2) * rot_x_kernels + self.psi.unsqueeze(2).unsqueeze(2))

        return F.conv2d(input, gabor_kernels, self.bias, self.stride, self.padding, self.dilation, self.groups)

    # In[4]: Compute the KL divergence losses. MUST keep the attribute name 'kl_loss' (see 'misc.py')
    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.psi_mu, self.psi_sigma)
        
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
