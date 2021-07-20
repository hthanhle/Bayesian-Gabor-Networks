import os
import numpy as np
from PIL import Image
import torch
import re
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset


class CityScapes(Dataset):
    NUM_CLASS = 19
    BASE_DIR = 'Cityscapes'

    def __init__(self, root='./public_datasets', split='train', img_size=(512, 1024)):
        super(CityScapes, self).__init__()

        self.resize_img = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.resize_mask = T.Resize(img_size, interpolation=Image.NEAREST)
        self.split = split
        root = os.path.join(root, self.BASE_DIR)
        ignore_label = -1
        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}
        self.images, self.masks = get_city_pairs(root, self.split)
        print("Loaded Cityscapes: {} subset with {} images".format(self.split, self.__len__()))

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.resize_img(img)
        img = T.ToTensor()(img)

        mask = Image.open(self.masks[index])
        mask = self.resize_mask(mask)
        mask = np.array(mask).astype('int32')
        mask = self.convert_label(mask)
        mask[mask == 255] = -1
        mask = torch.from_numpy(mask).squeeze().long()

        return img, mask

    def __len__(self):
        return len(self.images)


def get_city_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split(' ', line)
                imgpath = f"{folder}{ll_str[0].rstrip()}"
                maskpath = f"{folder}{ll_str[1].rstrip()}"
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        split_f = os.path.join(folder, 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths


def get_dataloader(img_size=(512, 1024),
                   batch_size=8,
                   num_workers=6,
                   dataset_dir='./public_datasets'):
    # Train dataloader
    train_dataset = CityScapes(root=dataset_dir, split='train', img_size=img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    # Train dataloader
    val_dataset = CityScapes(root=dataset_dir, split='val', img_size=img_size)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    # Make sure there are all from different datasets
    assert len(train_dataset) != len(val_dataset)
    return train_loader, val_loader, len(train_dataset)  # return len(train) for computing the loss
