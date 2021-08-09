import collections
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class PedestrianLaneDatasetWithFold(Dataset):
    def __init__(self, dataset_dir, subset='train', img_size=(320, 320), fold=1):
        """
        :param dataset_dir: root directory containing the dataset
        :param subset: subset which we are working with (e.g., 'train', 'test', and 'val')
        :param img_size: image size
        :param fold: cross-validation fold which we are working with
        """
        super(PedestrianLaneDatasetWithFold, self).__init__()
        self.filenames = collections.defaultdict(list)
        self.img_size = img_size
        self.resize_raw = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.resize_gt = T.Resize(img_size, interpolation=Image.NEAREST)

        self.to_tensor = T.ToTensor()
        self.train_test_val = subset
        self.fold = fold
        self.data_path = dataset_dir + '/images/'

        name_filepath = "{}/{}_fold{}.txt".format(dataset_dir, subset, fold)
        with open(name_filepath, 'r') as f:
            self.filenames = f.read().splitlines()
        print("Loaded PLVP3 dataset: Fold {}, {} subset with {} images".format(fold, subset, self.__len__()))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(self.data_path + filename + '.jpg')

        try:
            gt = Image.open(self.data_path + filename + '_lane.png')
        except:
            gt = Image.open(self.data_path + filename + '_lane.PNG')

        # Resize image to the desired size
        if self.img_size is not None:
            img = self.resize_raw(img)
            gt = self.resize_gt(gt)

        # Convert both to tensor
        img = self.to_tensor(img)
        gt = self.to_tensor(gt)
        gt = gt.squeeze().long()
        return img, gt


def get_dataloader(fold=1, img_size=(320, 320), batch_size=8, num_workers=6,
                   dataset_dir='/data/lane_segmentation/lane_dataset/05-PLVD'):
    """
    Get data loaders
    :param fold: cross-validation fold which we are working with
    :param img_size: image size
    :param batch_size: batch size
    :param num_workers: number of workers
    :param dataset_dir: root directory containing the dataset
    :return: training loader
             validation loader
             test loader
             length of the training loader (i.e., number of iterations)
    """
    # Train dataloader
    train_dataset = PedestrianLaneDatasetWithFold(dataset_dir, subset='train', img_size=img_size, fold=fold)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    # Validation dataloader
    val_dataset = PedestrianLaneDatasetWithFold(dataset_dir, subset='valid', img_size=img_size, fold=fold)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Test dataloader
    test_dataset = PedestrianLaneDatasetWithFold(dataset_dir, subset='test', img_size=img_size, fold=fold)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False,  # fix the filenames in order
                                              num_workers=num_workers)

    assert len(train_dataset) != len(val_dataset) != len(test_dataset)
    return train_loader, val_loader, test_loader, len(train_dataset)  # return len(train) for computing the loss
