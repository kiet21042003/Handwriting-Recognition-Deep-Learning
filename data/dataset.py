import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import csv

import torch
from torch.utils.data import Dataset
from data.utils import resize_image, normalize_tensor_image, padding

class HWDataset(Dataset):
    """
    Hand Writtten dataset. This class is used for training baseline.
    It will be used with collate_fn function
    """

    def __init__(self, root_dir, label_file, min_size, max_size, transform=None):
        """
        Arguments:
            label_file (string): Path to the label file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(HWDataset, self).__init__()
        self.labels = pd.read_csv(label_file, sep='\t', header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.min_size = min_size
        self.max_size = max_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx,0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform.forward(image)
            
        np_image = np.array(image)
        if np_image.ndim > 3:   # remove alpha channel
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(np_image)
        np_image = np_image.transpose((2,0,1))
            
        # image = np.array(image)
        # if image.ndim > 3:
        #     image = image[:, :, :3] # remove alpha channel if necessary
        label = str(self.labels.iloc[idx, 1]).strip()
        return np_image, label, self.min_size, self.max_size
        
class HWDatasetAugment(Dataset):    #TODO: Finish this augmentation dataset (augment in utils)
    """
    Hand Writtten dataset. This class is used for finetuning with augmented dataset.
    It will not be used with collate_fn function
    """

    def __init__(self, root_dir, label_file, transform=None):
        """
        Arguments:
            label_file (string): Path to the label file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(HWDataset, self).__init__()
        self.labels = pd.read_csv(label_file, sep='\t', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx,0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform.forward(image)
            
        np_image = np.array(image)
        if np_image.ndim > 3:   # remove alpha channel
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(np_image)
        np_image = np_image.transpose((2,0,1))
            
        # image = np.array(image)
        # if image.ndim > 3:
        #     image = image[:, :, :3] # remove alpha channel if necessary
        label = str(self.labels.iloc[idx, 1]).strip()
        return np_image, label
        
    