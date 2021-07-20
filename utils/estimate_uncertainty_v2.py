"""
Created on Thu Jul 16 11:24:09 2020
@author: Thanh Le
Estimate uncertainty using vectorization
"""

import torch
import numpy as np
from torch.nn import functional as F


def estimate_uncertainty(model, image, num_trials=15,
                         height=320, width=320, normalization=True):
    """

    :param model: trained Bayesian model
    :param image: input image as a Pytorch tensor
    :param num_trials: number of repetitive predictions
    :param height: original resolution rather than the network input shape of 320
    :param width: original resolution rather than the network input shape of 320
    :param normalization: softplus normalization
    :return: epistemic uncertainty map of size [H, W]
             aleatoric unertainty map of size [H, W]
             average logits (i.e., raw outputs before softmax) of size [num_classes, H, W]
             predicted scores (i.e., outputs after softmax) of size [num_trials, num_classes, H, W]
             predicted labels of size [num_trials, H, W]
             average segmentation map of size [H, W]
    """
    image = image.repeat(num_trials, 1, 1, 1)  # repeat the tensor along the first dimension: [1, 3, height, width] --> [num_trials, 3, height, width]

    logits, _ = model(image)  # perform a forward pass. Tensor shape: [num_trials, num_classes, height, width]
    logits = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(
        logits)  # unsampling visualizes the output at the original resolution but takes longer time, so it affects the inference time
    pred_scores = F.log_softmax(logits,
                                dim=1)  # perform the activation of log softmax. Expected Tensor shape: [1, num_class, height, width]

    # Calculate the epistemic/aleatoric unertainties for the input image. See the paper 'Uncertainty estimation by softplus normalization...'      
    if normalization:
        # Softplus normalization, see Eq. 14
        prediction = F.softplus(pred_scores)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(pred_scores, dim=1)  # should double-check since we perform the softmax activation twice

    p_hat = p_hat.detach().cpu().numpy()  # Expected Numpy array shape: [num_trials, num_classes, height, width]
    p_bar = np.mean(p_hat,
                    axis=0)  # calculate the 'p_bar', see Eq. 15. Expected Numpy array shape: [num_classes, height, width]

    # Estimate the epistemic unertainty, see Eq. 15
    subtract = p_hat - np.expand_dims(p_bar, 0)

    # Solution 1:
    epistemic = (subtract * subtract).sum(
        0) / num_trials  # Expected Numpy array shape: [num_classes, height, width]. So amazing with vectorization! see my Stackoverflow question

    # # Solution 2:
    # subtract_T_1 = np.transpose(subtract, (2,3,0,1))           # Shape (320,320,15,2)
    # subtract_T_2 = np.transpose(subtract, (2,3,1,0))           # Shape (320,320,2,15)
    # epistemics_new = (subtract_T_2 @ subtract_T_1)/ num_trials # Shape (320,320,2,2)
    # epistemics_new = np.diagonal(epistemics_new, axis1=2, axis2=3) # Shape (320,320,2)
    # epistemics_new = np.transpose(epistemics_new, (2,0,1))         # Shape (2,320,320)

    # # Solution 3:
    # epistemics_new = np.einsum('ijkl,ijkl->jkl',subtract,subtract)/ num_trials

    # Estimate the aleatoric unertainty, see Eq. 15    
    aleatoric = p_bar - (p_hat * p_hat).sum(
        0) / num_trials  # Expected Numpy array shape: [num_classes, height, width]. See how to vectorize the computation! see my Stackoverflow question
    # np.allclose(aleatoric_old,aleatoric_new, atol=1e-06)  #use this to compare the vectorized computation with the old one. Note that the tolerance is slightly higher than that of the epistemic uncertainty. That means the results before and after vectorization are a bit different, but still acceptable      

    epistemic = np.max(epistemic, axis=0)  # Expected Numpy array shape: [height, width]
    aleatoric = np.max(aleatoric, axis=0)

    pred_scores = pred_scores.cpu().detach().numpy()  # Detach the data from graph and convert to a Numpy array. Expected shape: [num_trials, num_classes, height, width]
    pred_labels = np.argmax(pred_scores,
                            axis=1)  # Get the predicted labels. Expected shape: [num_trials, height, width]

    # Get the mean of the predicted labels as the final segmentation map. Note that rounding the results to zero decimal places is equivalent to majority voting, because we only have 2 classes (expected shape: [height, width])   
    avg_segmentation_map = np.round(np.mean(pred_labels, axis=0), 0).astype(int)

    return epistemic, aleatoric, np.mean(logits.detach().cpu().numpy(),
                                         axis=0), pred_scores, pred_labels, avg_segmentation_map  # all outputs are Numpy arrays for 'compute_ece_ause_numpy.py'
