import numpy as np
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, f1_score
from torch import nn
import torch
from scipy.stats import norm


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl_loss, beta):
        assert not target.requires_grad
        nll_loss = F.nll_loss(input, target, reduction='mean', ignore_index=-1)

        # loss = KL_loss + NLL_loss.
        # Note that the 'nll_loss' is computed based on the network outputs (vs. ground-truths), while the 'kl_loss' is
        # accumulated over the Bayesian layers.
        return nll_loss * self.train_size + beta * kl_loss


def accuracy(seg_map, gt):
    """
    Calculate pixel accuracy
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: pixel accuracy
    """
    return np.mean(seg_map == gt)


def iou(seg_map, gt):  # inputs are Numpy arrays
    """
    Calculate mean IoU (a.k.a., Jaccard Index) of an individual segmentation map. Note that, for the whole dataset, we
    must take average the mIoUs
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: mIoU
    """
    # 'micro': Calculate metrics globally by counting the total TP, FN, and FP
    # 'macro': calculate metrics for each label, and find their unweighted mean.
    return jaccard_score(gt.flatten(), seg_map.flatten(), average='macro')


def f1_score(seg_map, gt):
    """
    Calculate F-measure (a.k.a., Dice Coefficient, F1-score)
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: F-measure
    """
    return f1_score(gt.flatten(), seg_map.flatten(), average="macro")


# In[3]: OPTION 1 (for BayesianGabor-v1/2): Compute the KL divergence between two distributions described by sigmas and mus
def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


# In[4]: OPTION 2 (for BayesianGabor-v3/4)
def calculate_kl_loss(w, mu, sigma, prior_sigma1=1.5, prior_sigma2=0.1):
    # Convert all input tensors to Numpy arrays
    w = w.cpu().detach().numpy()
    mu = mu.cpu().detach().numpy()
    sigma = sigma.cpu().detach().numpy()

    log_variational = norm.logpdf(w, loc=mu, scale=sigma)
    log_prior = np.log(0.5 * norm.pdf(w, loc=0, scale=prior_sigma1) +
                       0.5 * norm.pdf(w, loc=0, scale=prior_sigma2))

    kl_loss = np.sum(log_variational - log_prior)
    return kl_loss


# In[5]:Get KL weight
def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
