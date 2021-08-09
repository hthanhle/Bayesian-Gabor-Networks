"""
Estimate uncertainty
@author: Thanh Le
"""

import torch
import numpy as np
from torch.nn import functional as F


def estimate_uncertainty(model, image, height, width, num_trials=15, normalization=True):
    """
    Estimate uncertainty
    :param model: trained Bayesian model
    :param image: input image as a Pytorch tensor
    :param height: original resolution
    :param width: original resolution
    :param num_trials: number of repetitive predictions
    :param normalization: softplus normalization
    :return: epistemic uncertainty map of size [H, W]
             aleatoric unertainty map of size [H, W]
             average logits (i.e., raw outputs before softmax) of size [num_classes, H, W]
             predicted scores (i.e., outputs after softmax) of size [num_trials, num_classes, H, W]
             predicted labels of size [num_trials, H, W]
             average segmentation map of size [H, W]
    """
    image = image.repeat(num_trials, 1, 1, 1)  # repeat the tensor along the first dimension

    logits, _ = model(image)  # perform a forward pass

    # Upsample the output at the original resolution (for a better visualization)
    logits = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(logits)
    pred_scores = F.log_softmax(logits, dim=1)

    if normalization:
        prediction = F.softplus(pred_scores)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(pred_scores, dim=1)

    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    # Estimate the epistemic uncertainty
    subtract = p_hat - np.expand_dims(p_bar, 0)
    epistemic = (subtract * subtract).sum(0) / num_trials
    epistemic = np.max(epistemic, axis=0)

    # Estimate the aleatoric uncertainty
    aleatoric = p_bar - (p_hat * p_hat).sum(0) / num_trials
    aleatoric = np.max(aleatoric, axis=0)

    # Get the predicted labels
    pred_scores = pred_scores.cpu().detach().numpy()
    pred_labels = np.argmax(pred_scores, axis=1)

    # Get the mean of the predicted labels as the final segmentation map
    avg_segmentation_map = np.round(np.mean(pred_labels, axis=0), 0).astype(int)

    return epistemic, aleatoric, np.mean(logits.detach().cpu().numpy(),
                                         axis=0), pred_scores, pred_labels, avg_segmentation_map
