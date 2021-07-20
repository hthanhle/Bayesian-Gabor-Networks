import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T


class PascalVOC12(Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    BASE_DIR = 'PascalVOC12/VOCdevkit/VOC2012'

    def __init__(self, root='./public_datasets', split='train', img_size=(480, 480)):
        super(PascalVOC12, self).__init__()

        self.root = root
        self.split = split
        self.resize_img = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.resize_mask = T.Resize(img_size, interpolation=Image.NEAREST)

        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        _split_f = os.path.join(_splits_dir, self.split + '.txt')

        # Get the file list of images and ground-truths
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        print("Loaded PascalVOC12: {} subset with {} images".format(self.split, self.__len__()))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.resize_img(img)
        img = T.ToTensor()(img)

        mask = Image.open(self.masks[index])
        mask = self.resize_mask(mask)
        mask = np.array(mask).astype('int32')
        mask[mask == 255] = -1
        mask = torch.from_numpy(mask).squeeze().long()

        return img, mask

    def __len__(self):
        return len(self.images)


def get_dataloader(img_size=(512, 1024),
                   batch_size=8,
                   num_workers=6,
                   dataset_dir='./public_datasets'):
    # Train dataloader
    train_dataset = PascalVOC12(root=dataset_dir, split='train', img_size=img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    # Train dataloader
    val_dataset = PascalVOC12(root=dataset_dir, split='val', img_size=img_size)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    # Make sure there are all from different datasets
    assert len(train_dataset) != len(val_dataset)
    return train_loader, val_loader, len(train_dataset)  # return len(train) for computing the loss