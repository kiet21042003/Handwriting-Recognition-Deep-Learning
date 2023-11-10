import os
import numpy as np
import cv2
import pandas as pd
import csv

import torch
from torch.utils.data import Dataset
from data.utils import resize_image, normalize_tensor_image, padding

class HWDataset(Dataset):
    '''
    Handwriting English Dataset
    '''
    def __init__(
        self, 
        root_dir: str,
        max_size: int,
        min_size: int,        
        max_len: int,
        split_type='A',
        mode='train',
    ):
        """Initialization of HW Dataset

        Args:
            root_dir (str): Relative root directory of images
            label_file (str): Relative path to txt file containing all labels and corresponding image names
            mode (str, optional): Mode of dataset. Defaults to 'train'.
            preprocess (bool, optional): Whether to perform preprocessing
            max_len (int, optional): Maximum length of the label
        """

        self.root_dir = os.path.join(root_dir, mode)
        self.image_dir = os.path.join(self.root_dir, 'data')
        self.label_file = os.path.join(self.root_dir, 'label.csv')
        
        INVALIDS = ['a01-117-05-02.png', 'r06-022-03-05.png']   # samples that we found corrupted 
        
        old_data_dict = pd.read_csv(
                        self.label_file, sep="\t",
                        header=0, encoding="utf-8", 
                        na_filter=False, engine="python").to_dict(orient='index')
        
        self.data_dict = dict()
        cnt = 0
        for k in old_data_dict:
            label = old_data_dict[k]
            if label['Image'] not in INVALIDS:
                self.data_dict[cnt] = label
                cnt += 1
        
        self.mode = mode
        self.max_len = max_len
        self.max_size = max_size
        self.min_size = min_size
        
        assert(self.mode in ['train', 'test', 'val']), f"Mode must be train, test or val. The given mode value is {self.mode}"
    
    def get_item_name(self, idx):
        img_name = self.data_dict[idx]['Image'].strip()
        return img_name
            
    def __len__(self):
        return len(self.data_dict.keys())
    
    def __getitem__(self, idx):
        img_name = self.data_dict[idx]['Image'].strip()
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = image.astype("float32")
        
        if image.ndim==2:
            image = image[np.newaxis]
        image = image.transpose((2,0,1))
            
        # Label getter
        label = self.data_dict[idx]['Label'].strip()
        return image, label, self.min_size, self.max_size
        
            
            
        
        
    