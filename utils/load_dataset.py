import collections
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np


class PedestrianLaneDatasetWithFold(Dataset):
    def __init__(self,
                 dataset_dir,
                 train_test_val='train',
                 img_size=(320, 320),
                 fold=1):
        """
        Args:
            dataset_dir (string): Directory containing train, test and val folders.
                               E.g. each folder contains raw and mask folders.
            train_test_val (string): Which subset to use? (train/test/val)
            img_size (tuple): H and W of the image to be resized to.
        """
        super(PedestrianLaneDatasetWithFold, self).__init__()
        self.filenames = collections.defaultdict(list)
        self.img_size = img_size
        self.resize_raw = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.resize_gt = T.Resize(img_size, interpolation=Image.NEAREST)

        self.to_tensor = T.ToTensor()
        self.train_test_val = train_test_val
        self.fold = fold
        self.data_path = dataset_dir + '/images/'  # Path to the images. IMPORTANT: assume that all images and ground-truths are in the folder 'images'

        name_filepath = "{}/{}_fold{}.txt".format(dataset_dir, train_test_val, fold)
        with open(name_filepath, 'r') as f:
            self.filenames = f.read().splitlines()
        print("Loaded PLVP3 dataset: Fold {}, {} subset with {} images".format(fold, train_test_val, self.__len__()))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(self.data_path + filename + '.jpg')

        # Some ground-truth has .PNG extension instead of small caps.
        try:
            gt = Image.open(self.data_path + filename + '_lane.png')
        except:
            gt = Image.open(self.data_path + filename + '_lane.PNG')

        # Resize image to the desired size
        if self.img_size is not None:
            img = self.resize_raw(img)  # Resizing is done directly here, the caller does not need to use a "Transform"
            gt = self.resize_gt(gt)

        # Convert both to tensor
        img = self.to_tensor(img)  # Tensor shape: [3, H, W]. Note that, when reading as a batch, the Tensor shape should be [batch, 3, H, W]
        gt = self.to_tensor(gt)
        gt = gt.squeeze().long()  # Tensor shape: [H, W]. Note that, when reading as a batch, the Tensor shape should be [batch, H, W]
        return img, gt


def get_dataloader(fold=1,
                   img_size=(320, 320),
                   batch_size=8,
                   num_workers=6,
                   dataset_dir='/data/lane_segmentation/lane_dataset/05-PLVD'):

    # Train dataloader
    train_dataset = PedestrianLaneDatasetWithFold(dataset_dir, train_test_val='train', img_size=img_size, fold=fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    # Validation dataloader
    val_dataset = PedestrianLaneDatasetWithFold(dataset_dir, train_test_val='valid', img_size=img_size, fold=fold)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Test dataloader
    test_dataset = PedestrianLaneDatasetWithFold(dataset_dir, train_test_val='test', img_size=img_size, fold=fold)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False,  # fix the filenames in order
                                              num_workers=num_workers)

    # Make sure there are all from different datasets
    assert len(train_dataset) != len(val_dataset) != len(test_dataset)
    return train_loader, val_loader, test_loader, len(train_dataset)  # return len(train) for computing the loss
