"""
Created on Fri July 09 11:54:53 2021
Dependencies: Pytorch 1.2.0
@author: Thanh Le
"""
import torch
from tqdm import tqdm
import numpy as np
from utils.estimate_uncertainty_v2 import estimate_uncertainty
from timeit import default_timer as timer
import yaml
from torch.nn import functional as F
from sklearn.metrics import jaccard_score, f1_score
from utils.compute_ece_ause_numpy import ECEHelper, AUSEHelper
# Turn off the warning of YAML loader
import warnings

warnings.filterwarnings('ignore')


# CUDA settings
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


def test_model(model, test_loader):
    model.eval()
    inference_time = []
    f_measures = []
    accs = []
    ious = []
    ece_helpers = ECEHelper(n_bins=10)
    ause_helpers = AUSEHelper()

    for i, (img, gt) in enumerate(tqdm(test_loader, ncols=80, desc='Testing')):
        # Get img and gt (i.e. a batch), then send them to the GPU every step. 'to(device)' is used to move data to GPU
        img = img.to(device,
                     dtype=torch.float)  # expected Tensor shape: img of size [batch_size, 3, H, W], gt of size [batch_size, H, W]. Note that 'batch_size' is always 1 for testing
        gt = gt.cpu().detach().numpy()  # 'gt' is Numpy array, no longer a Tensor. We do not need to move the ground-truth to GPU, since it is only used for evaluation

        start = timer()
        pred_score, _ = model(img)  # Tensor shape: [batch_size, num_classes, H, W]. The reason we perform this single forward pass is just for measuring the inference time
        end = timer()
        pred_score = F.log_softmax(pred_score, dim=1)  # softmax activation. Tensor shape: [batch_size, num_class, H, W]
        inference_time.append(end - start)

        # pred_score = pred_score.squeeze(0)  # remove the first dimension of 1
        # pred_score = pred_score.cpu().detach().numpy()  # Shape: [num_class, H, W]
        # seg_map = np.argmax(pred_score, axis=0)

        # Estimate the uncertainty maps. Note that 'seg_map' is the averaged map, and a bit different from the
        # segmentation map in line 58.
        epistemic, aleatoric, logits, _, _, avg_seg_map = estimate_uncertainty(model, img, num_trials=15)

        # Accumulate the performance using Scikit-learn
        accs.append(np.mean(avg_seg_map == gt))
        ious.append(jaccard_score(gt.flatten(), avg_seg_map.flatten(), average='macro'))
        f_measures.append(f1_score(gt.flatten(), avg_seg_map.flatten(), average='macro'))

        # Compute ECE and AUSE
        logits = np.expand_dims(logits, axis=0)  # Shape: [1, num_classes, H, W]
        uncertainty_map = np.expand_dims(aleatoric, axis=0)  # Shape: [1, H, W]
        ece_helpers.distribute_to_bins(logits=logits, labels=gt)  # Accumulate the values
        ause_helpers.store_values(logits=logits, labels=gt, uncertainty=uncertainty_map)

    return np.mean(accs), np.mean(ious), np.mean(f_measures), np.mean(inference_time[
                                                                      10:]), ece_helpers.get_ece(), ause_helpers.get_ause()  # skip several first predictions so that the inference time becomes stable


if __name__ == '__main__':
    # 1. Setup CUDA
    device = setup_cuda()

    # 2. Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file)

    # 3. Load the configurations
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

    # 4. Load the dataset
    from utils.load_dataset import get_dataloader

    _, _, test_loader, _ = get_dataloader(fold=fold, img_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          dataset_dir=dataset_dir)

    # 5. Create a new Bayesian model, then load the pre-trained weights
    # OPTION 1: Bayesian Gabor Network
    # from models.BayesianGaborNetwork_v1 import BayesianGaborNetwork
    # model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianGaborNetwork_v1'

    # OPTION 2: Bayesian Gabor Network
    from models.BayesianGaborNetwork_v2 import BayesianGaborNetwork
    model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    model_name = 'BayesianGaborNetwork_v2'

    # OPTION 3: Bayesian Gabor Network v3
    # from models.BayesianGaborNetwork_v3 import BayesianGaborNetwork
    # model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianGaborNetwork_v3'

    # OPTION 4: Bayesian CNN
    # from models.BayesianCNN import BayesianCNN
    # model = BayesianCNN(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianCNN'

    model.load_state_dict(torch.load(weight_path, device))
    from torchsummary import summary  # pip install torch-summary
    summary(model, input_size=(3, img_height, img_width))
    print('Loading ' + weight_path + ' done')

    # Evaluate the model
    acc, miou, f1, time, ece, ause = test_model(model, test_loader)

    print(
        'Accuracy: {}. mIoU: {}. F-measure: {}. Inference time: {}. ECE: {}. AUSE: {}'.format(acc, miou, f1, time, ece,
                                                                                              ause))
