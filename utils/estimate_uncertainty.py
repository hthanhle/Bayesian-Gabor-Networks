"""
Created on Thu Jul 16 11:24:09 2020
@author: Thanh Le
Estimate uncertainty using naive for loops
"""

import torch
import numpy as np
from torch.nn import functional as F


# Given a trained model and a SINGLE input image, we make predictions several times then return uncertainties,the logits (before softmax), the predicted scores (after softmax), the predicted labels and the averaged segmentation map
def estimate_uncertainty(model, image, device, num_classes=2, num_trials=15,
                         height=320, width=320,
                         normalization=True):  # Note that 'image' MUST be either a CV2 image (when inputing a single image) or a Pytorch tensor (when enumerating whole dataset)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)  # convert to a Pytorch tensor if 'image' is a Numpy array
        image = image.unsqueeze(
            0)  # expand the tensor along the first dimension (i.e. batch size): [3, height, width] --> [1, 3, height, width]

    image = image.repeat(num_trials, 1, 1,
                         1)  # repeat the tensor along the first dimension: [1, 3, height, width] --> [num_trials, 3, height, width]
    image = image.to(device, dtype=torch.float)  # move the data to the GPU

    logits, _ = model(
        image)  # get the outputs after a forward pass. Expected Tensor shape: [num_trials, num_classes, height, width]
    pred_scores = F.log_softmax(logits, dim=1)  # perform the activation of log softmax

    # Calculate the epistemic/aleatoric unertainties for the input image. See the paper 'Uncertainty estimation by softplus normalization...'      
    if normalization:
        # Softplus normalization, see Eq. 14
        prediction = F.softplus(pred_scores)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(pred_scores, dim=1)

    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)  # calculate the 'p_bar', see Eq. 15

    # Estimate the epistemic unertainty, see Eq. 15
    subtract = p_hat - np.expand_dims(p_bar, 0)
    epistemics = np.zeros((num_classes, height, width))
    for row in range(height):
        for col in range(width):
            epistemic = np.dot(subtract[:, :, row, col].T, subtract[:, :, row, col]) / num_trials
            epistemic = np.diag(epistemic)
            epistemics[:, row, col] = epistemic

    # Estimate the aleatoric unertainty, see Eq. 15
    aleatorics = np.zeros((num_classes, height, width))
    for row in range(height):
        for col in range(width):
            aleatoric = np.diag(p_bar[:, row, col]) - (
                        np.dot(p_hat[:, :, row, col].T, p_hat[:, :, row, col]) / num_trials)
            aleatoric = np.diag(aleatoric)
            aleatorics[:, row, col] = aleatoric

    epis = np.max(epistemics, axis=0)
    aleas = np.max(aleatorics, axis=0)

    pred_scores = pred_scores.cpu().detach().numpy()  # Detach the data from graph and convert to a Numpy array. Expected shape: [num_trials, num_classes, height, width]
    pred_labels = np.argmax(pred_scores,
                            axis=1)  # Get the predicted labels. Expected shape: [num_trials, height, width]

    # Get the mean of the predicted labels as the final segmentation map. Note that rounding the results to zero decimal places is equivalent to majority voting, because we only have 2 classes (expected shape: [height, width])   
    avg_segmentation_map = np.round(np.mean(pred_labels, axis=0), 0).astype(int)

    return epis, aleas, np.mean(logits.detach().cpu().numpy(),
                                axis=0), pred_scores, pred_labels, avg_segmentation_map  # all outputs are Numpy arrays for 'compute_ece_ause_numpy.py'
