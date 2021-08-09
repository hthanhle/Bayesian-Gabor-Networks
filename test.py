"""
Created on Fri July 09 11:54:53 2021
Test a pre-trained Bayesian Gabor Network
@author: Thanh Le
"""
import torch
from tqdm import tqdm
import numpy as np
from utils.estimate_uncertainty_v2 import estimate_uncertainty
from timeit import default_timer as timer
import yaml
from torchsummary import summary
from torch.nn import functional as F
from sklearn.metrics import jaccard_score, f1_score
from utils.compute_ece_ause import ECEHelper, AUSEHelper
import warnings

warnings.filterwarnings('ignore')


def setup_cuda():
    # Setting seeds
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model():
    model.eval()
    inference_time = []
    f_measures = []
    accs = []
    ious = []
    ece_helpers = ECEHelper(n_bins=10)
    ause_helpers = AUSEHelper()

    for i, (img, gt) in enumerate(tqdm(test_loader, ncols=80, desc='Testing')):
        img = img.to(device, dtype=torch.float)
        gt = gt.cpu().detach().numpy()

        start = timer()
        pred_score, _ = model(img)
        end = timer()
        pred_score = F.log_softmax(pred_score, dim=1)
        inference_time.append(end - start)

        # Estimate uncertainties
        epistemic, aleatoric, logits, _, _, avg_seg_map = estimate_uncertainty(model, img, num_trials=15)
        accs.append(np.mean(avg_seg_map == gt))
        ious.append(jaccard_score(gt.flatten(), avg_seg_map.flatten(), average='macro'))
        f_measures.append(f1_score(gt.flatten(), avg_seg_map.flatten(), average='macro'))

        # Compute ECE and AUSE
        logits = np.expand_dims(logits, axis=0)
        uncertainty_map = np.expand_dims(aleatoric, axis=0)
        ece_helpers.distribute_to_bins(logits=logits, labels=gt)
        ause_helpers.store_values(logits=logits, labels=gt, uncertainty=uncertainty_map)

    return np.mean(accs), np.mean(ious), np.mean(f_measures), \
           np.mean(inference_time[10:]), ece_helpers.get_ece(), ause_helpers.get_ause()


if __name__ == '__main__':
    # Setup CUDA
    device = setup_cuda()

    # Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file)

    # Load the configurations
    dataset_dir = cfg['train_params']['dataset_dir']
    fold = cfg['train_params']['fold']
    num_inputs = cfg['train_params']['num_inputs']
    num_outputs = len(cfg['train_params']['classes'])
    weight_path = cfg['train_params']['weight_path']
    batch_size = 1  # for testing
    num_workers = cfg['train_params']['num_workers']
    img_height = cfg['train_params']['img_height']
    img_width = cfg['train_params']['img_width']
    activation_type = cfg['model_params']['activation_type']
    beta_type = cfg['bayes_params']['beta_type']
    priors = cfg['bayes_params']['priors']

    # Load the dataset
    from utils.load_dataset import get_dataloader

    _, _, test_loader, _ = get_dataloader(fold=fold, img_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          dataset_dir=dataset_dir)

    # Create a new Bayesian model, then load the pre-trained weights
    from models.BayesianGaborNetwork_v1 import BayesianGaborNetwork

    model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation=activation_type).to(device)
    model.load_state_dict(torch.load(weight_path, device))
    summary(model, input_size=(3, img_height, img_width))

    # Evaluate the model
    acc, miou, f1, time, ece, ause = test_model()
    print('Acc: {}. mIoU: {}. F1: {}. Pred time: {}. ECE: {}. AUSE: {}'.format(acc, miou, f1, time, ece, ause))
