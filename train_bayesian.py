"""
Created on Feb 06 15:53:16 2021
Dependencies: Pytorch 1.2.0
@author: Thanh Le
Train a Bayesian CNN or a Bayesian Gabor Network
"""
import torch
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from torch.nn import functional as F
from utils.utils import logmeanexp
from utils import metrics
import yaml
import json
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


# Train the model (over a single epoch)
def train_model(num_ens=1):
    # Tell the model that you are training. So effectively layers (e.g. dropout, batch-norm)
    # which behave different on the train and test procedures know what is going on 
    # and hence can behave accordingly
    model.train()

    training_loss = 0.0
    performance = 0
    p_bar = tqdm(train_loader, ncols=80, desc='Training')

    for i, (img, gt) in enumerate(p_bar):  # loads data every iteration
        # Set the gradients to zero before starting backpropagation because
        # PyTorch accumulates the gradients on subsequent backward passes. This
        # is convenient while training RNNs. So, the default action is to
        # accumulate (i.e. sum) the gradients on every loss.backward() call.
        optimizer.zero_grad()

        # Get images and ground-truths (i.e. a batch), then send them to the device every step
        # Tensor shape: img of size (batch_size, 3, H, W), gt of size (batch_size, H, W)
        img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)

        # 3. Perform a forward pass
        outputs = torch.zeros(img.shape[0], num_outputs,  # 2 classes
                              img.size()[2], img.size()[3], num_ens).to(
            device)  # Note that 'images.shape[0]' is the batch size
        kl = 0.0

        for j in range(num_ens):  # num_ens should be the number of predictions for each input image
            # Forward pass
            net_out, _kl = model(img)  # Tensor shape: (batch_size, num_classes, H, W)
            kl += _kl
            outputs[:, :, :, :, j] = F.log_softmax(net_out,
                                                   dim=1)  # softmax activation. Tensor shape: (batch_size, num_class, H, W, num_ens)

        kl = kl / num_ens

        log_outputs = logmeanexp(outputs, dim=4)

        # Calculate the loss
        beta = metrics.get_beta(i - 1, len(train_loader), beta_type, epoch,
                                num_epochs)  # 'beta' is the weight (i.e. coefficient) for the KL loss, see Eq. 9 in the paper 'Weight Uncertainty...'

        loss = loss_fn(log_outputs, gt, kl, beta)  # the loss function

        # Compute gradient of loss w.r.t all the trainable parameters (i.e. requires_grad = True)
        # in loss, and store them in the attribute 'parameter.grad' for every parameter
        loss.backward()

        # Updates all the trainable parameters based on 'parameter.grad'
        optimizer.step()

        # Accumulate the batch loss
        training_loss += loss.item()

        # 7. Accumulate the performance for every iteration
        # OPTION 1: Use Scikit-learn
        seg_maps = log_outputs.cpu().detach().numpy().argmax(axis=1)
        gt = gt.cpu().detach().numpy()
        performance += getattr(metrics, metric)(seg_maps, gt)  # getattr(metrics, 'accuracy') --> utils.metrics.accuracy

        # OPTION 2: Use Pytorch-lightning
        # seg_maps = torch.argmax(log_outputs, dim=1)
        # n_ext_classes = len(torch.unique(torch.stack((seg_maps, ground-truths), dim=0)))
        # performance += (getattr(plf, metric)(seg_maps, ground-truths, reduction='sum', num_classes=5) / n_ext_classes).item()

    return training_loss / len(train_loader), performance / len(train_loader)  # len(DataLoader) --> num of iterations


# Validate the model (over a single epoch). It is similar to the function 'train_model',
# except without computing the gradients (i.e. loss.backward) and updating the parameters (i.e. optimizer.step)
def validate_model(num_ens=1):
    model.eval()
    valid_loss = 0.0
    performance = 0

    with torch.no_grad():  # temporarily set all of the 'requires_grad' flags to 'false'. This aims to avoid CUDA out of memory
        for i, (img, gt) in enumerate(valid_loader):
            # Get images and ground-truths (i.e. a batch), then send them to the device every step
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)

            outputs = torch.zeros(img.shape[0], num_outputs,
                                  img.size()[2], img.size()[3], num_ens).to(
                device)  # Tensor shape: (batch_size, num_class, H, W, num_ens)
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = model(img)
                kl += _kl
                outputs[:, :, :, :, j] = F.log_softmax(net_out, dim=1)  # softmax activation

            log_outputs = logmeanexp(outputs, dim=4)

            # Accumulate the batch loss
            beta = metrics.get_beta(i - 1, len(valid_loader), beta_type, epoch, num_epochs)
            valid_loss += loss_fn(log_outputs, gt, kl, beta).item()

            # Accumulate the performance for every iteration
            # OPTION 1: Use Scikit-learn
            seg_maps = log_outputs.cpu().detach().numpy().argmax(axis=1)
            gt = gt.cpu().detach().numpy()
            performance += getattr(metrics, metric)(seg_maps,
                                                    gt)  # getattr(metrics, 'accuracy') --> utils.metrics.accuracy

            # OPTION 2: Use Pytorch-lightning
            # seg_maps = torch.argmax(log_outputs, dim=1)
            # n_ext_classes = len(torch.unique(torch.stack((seg_maps, ground-truths), dim=0)))
            # performance += (getattr(plf, metric)(seg_maps, ground-truths, reduction='sum', num_classes=5) / n_ext_classes).item()

    return valid_loss / len(valid_loader), performance / len(valid_loader)


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
    num_inputs = cfg['train_params']['num_inputs']  # 3 input RGB channels
    num_outputs = len(cfg['train_params']['classes'])  # 2 classes: background vs. lane
    num_epochs = cfg['train_params']['num_epochs']
    batch_size = cfg['train_params']['batch_size']
    num_workers = cfg['train_params']['num_workers']
    img_height = cfg['train_params']['img_height']
    img_width = cfg['train_params']['img_width']
    metric = cfg['model_params']['metric']
    lr_start = cfg['train_params']['lr_start']
    activation_type = cfg['model_params']['activation_type']
    beta_type = cfg['bayes_params']['beta_type']
    priors = cfg['bayes_params']['priors']  # for BayesianGabor_v3

    # 4. Load the dataset
    # OPTION 1: PLVP3
    # from utils.load_dataset import get_dataloader
    # train_loader, valid_loader, _, train_len = get_dataloader(fold=fold, img_size=(img_height, img_width),
    #                                                                     batch_size=batch_size, num_workers=num_workers,
    #                                                                     dataset_dir=dataset_dir)
    # ckp_path = './checkpoints/fold_' + str(fold) + '/'
    # perf_path = 'performance/fold_' + str(fold) + '/'

    # OPTION 2: Cityscapes
    # from utils.public_datasets.cityscapes import get_dataloader
    #
    # num_outputs = 19
    # img_height = 512
    # img_width = 1024
    # train_loader, valid_loader, train_len = get_dataloader(img_size=(img_height, img_width),
    #                                                        batch_size=batch_size,
    #                                                        num_workers=num_workers)
    # ckp_path = './checkpoints/Cityscapes/'
    # perf_path = './performance/Cityscapes/'

    # OPTION 3: PascalVOC12
    from utils.public_datasets.pascal_voc import get_dataloader
    num_outputs = 21
    img_height = 320
    img_width = 320
    train_loader, valid_loader, train_len = get_dataloader(img_size=(img_height, img_width),
                                                           batch_size=batch_size,
                                                           num_workers=num_workers)
    ckp_path = './checkpoints/PascalVOC12/'
    perf_path = './performance/PascalVOC12/'

    # 5. Create a Bayesian model
    # OPTION 1: BGN-v1
    from models.BayesianGaborNetwork_v1 import BayesianGaborNetwork
    model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    model_name = 'BayesianGaborNetwork_v1'

    # OPTION 2: BGN-v2
    # from models.BayesianGaborNetwork_v2 import BayesianGaborNetwork
    # model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianGaborNetwork_v2'

    # OPTION 3: BGN-v3
    # from models.BayesianGaborNetwork_v3 import BayesianGaborNetwork
    # model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianGaborNetwork_v3'

    # OPTION 3: BGN-v4
    # from models.BayesianGaborNetwork_v4 import BayesianGaborNetwork
    # model = BayesianGaborNetwork(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianGaborNetwork_v4'

    # OPTION 4: Bayesian CNN
    # from models.BayesianCNN import BayesianCNN
    # model = BayesianCNN(num_inputs, num_outputs, priors, activation_type=activation_type).to(device)
    # model_name = 'BayesianCNN'

    print('Beta type: {}. Model: {}'.format(beta_type, model_name))

    # Load the pre-trained weight if specified
    if cfg['train_params']['resumed']:
        model.load_state_dict(torch.load(cfg['train_params']['weight_path'], device))
        print('Loading the saved checkpoint done: {}. Training process resumed...'.format(cfg['train_params']['weight_path']))

    # 6. Specify loss function and optimizer
    loss_fn = metrics.ELBO(train_len).to(device)  # loss function
    optimizer = Adam(model.parameters(), lr=lr_start)  # optimizer    

    # 7. Start training the model
    max_perf = 0
    perf = []  # Store the validation performance for plotting the training behaviour
    for epoch in range(num_epochs):
        # 7.1. Train the model over a single epoch
        train_loss, train_perf = train_model()

        # 7.2. Validate the model
        test_loss, test_perf = validate_model()

        print('Epoch: {} \tTraining {}: {:.4f} \tValid {}: {:.4f}'.format(epoch, metric, train_perf, metric, test_perf))

        # 7.3. (Optional) Store the validation performance into a list. We immediately save the list so that we can stop the training process anytime.
        perf.append(test_perf)
        with open(perf_path + model_name + '_epoch_' + str(epoch) + '.txt', 'w') as f:
            json.dump(perf, f)

        # 7.4. Save the model if the validation performance is increasing
        if test_perf >= max_perf:
            print('Valid {} increased ({:.4f} --> {:.4f}). Model saved'.format(metric, max_perf, test_perf))
            torch.save(model.state_dict(), ckp_path + model_name + '_epoch_' + str(epoch) +
                       '_' + metric + '_{0:.4f}'.format(test_perf) + '.pt')
            max_perf = test_perf

        # 7.5 (Optional) Adjust the learning rate
        # if epoch == 100:
        #     adjust_learning_rate(optimizer, lr=lr_start / 10)  # lr of 0.0001
        #     print('Learning rate adjusted')
