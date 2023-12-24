import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import csv

import torch
from torch.utils.data import Dataset
from data.utils import preprocess

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
        image = cv2.imread(img_name).astype(np.float32)
        if image.ndim==2:
            image = image[np.newaxis]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = preprocess(image, self.min_size, self.max_size)
        # if self.transform:
        #     image = self.transform(image)
        label = str(self.labels.iloc[idx, 1]).strip()
        
        return image, label
        
