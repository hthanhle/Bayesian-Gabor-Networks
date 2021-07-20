# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:26:00 2020

@author: tlh857
"""
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import os
import torch.utils.data as data
from load_dataset import AutoLaneDataset
import torchvision.transforms as T
#transf = transforms.Compose([
#    transforms.Resize((288, 288)),
#    transforms.ToTensor(),
#    ])
#
## In[1]: Define a custom dataset class
#class CustomDataset(Dataset):
#    def __init__(self, images, groundtruths, train=True): 
#
#        self.images = images
#        self.groundtruths = groundtruths
#        self.transforms = transf
#
#    def __getitem__(self, index):
#        image        = Image.open(self.images[index])
#        groundtruth  = Image.open(self.groundtruths[index])
#        t_image      = self.transforms(image)
#        return t_image, groundtruth
#
#    def __len__(self):
#        return len(self.images)
#	
#	
## In[1]:	
#class DataLoaderSegmentation(data.Dataset):
#    def __init__(self, folder_path):
#        super(DataLoaderSegmentation, self).__init__()
#        self.img_files  = glob.glob(os.path.join(folder_path,'images','*.jpg'))        
#        self.mask_files = glob.glob(os.path.join(folder_path,'groundtruths','*.png'))
#        self.transforms = transf		
#
#    def __getitem__(self, index):
#            img_path  = self.img_files[index]
#            mask_path = self.mask_files[index]
#            data      = Image.open(img_path)
#            label     = Image.open(mask_path)
#            data      = self.transforms(data)
#            label     = self.transforms(label)
#            return np.asarray(data), np.asarray(label)
#
#    def __len__(self):
#        return len(self.img_files)
	

# In[1]:
class LaneDataset(Dataset):
    def __init__(self, images, groundtruths): # 4D Numpy arrays
        self.images       = images
        self.groundtruths = groundtruths
        self.transforms   = transforms.Compose([transforms.ToTensor(),
											  ])

    def __getitem__(self, index):
        image        = self.images[index, :, :, :]
        groundtruth  = self.groundtruths[index, :, :]        
        return (image, groundtruth)

    def __len__(self):
        return len(self.images)		
	

# In[2]: Read data and the corresponding groundtruth
#def read_segmentation_data_1(image_folder, groundtruth_folder, batch_size=4, num_workers=4, train=True):
#    # Get all the images and gounrdtruths in the folders
#    images       = glob.glob(image_folder)
#    groundtruths = glob.glob(groundtruth_folder)
#        
#    dataset      = CustomDataset(images, groundtruths, train=train)
#    data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#    
#    return dataset, data_loader

# In[2]:
#def read_segmentation_data_2(root_folder, num_workers=4, train=True):        
#    dataset      = DataLoaderSegmentation(root_folder)
#    data_loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=num_workers)
#    
#    return dataset, data_loader

# In[2]:
def read_segmentation_data(images, groundtruths, batch_size=4, num_workers=4):        
    dataset      = LaneDataset(images, groundtruths)
    data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset, data_loader


def get_training_dataloaders(cfg):
    '''
    Args:
        cfg: configuration dict
    Returns:
        - train, valid and test dataloaders
    '''
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    root_dir = cfg['dataset_dir']
    transform = T.Compose([
        T.Resize(img_size)
    ])
    # Train dataloaders
    train_dataset = AutoLaneDataset(root_dir + '/training/', train_test_val='training_9k', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # Test dataloader
    test_dataset = AutoLaneDataset(root_dir + '/validation/', train_test_val='validation', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # Make sure there are all from different datasets
    # assert len(train_dataset) != len(test_dataset)
    return train_loader, test_loader, len(train_dataset)

# In[3]: Main
# test_image_folder       = 'test/images/*.jpg'
# test_groundtruth_folder = 'test/groundtruths/*.png'

# dataset, data_loader = read_segmentation_data(test_image_folder, test_groundtruth_folder, train=True)
