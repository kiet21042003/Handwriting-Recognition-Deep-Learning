import torch
import torch.nn as nn
import cv2
import torchvision.transforms as tfs
import numpy as np

class Augmentation(object):
    def __init__(self, max_size, min_size):
        # assume H < W
        self.general_augmenter = tfs.Compose([
            tfs.RandomRotation(degrees=10, expand=True),
            tfs.ColorJitter(brightness=0.5, contrast=1.0, saturation=1.0),
            tfs.PILToTensor()
        ])
        self.max_size = max_size
        self.min_size = min_size
        
    def forward(self, img):
        return self.transform(img)

def preprocess(image: torch.tensor, min_size, max_size):
    _, H, W = image.size()
    image = resize_image(image, min_size=min_size, max_size=max_size)
    image = normalize_tensor_image(image)
    image = padding(image, min_size=min_size, max_size=max_size)
    return image

def collate_fn(data):
    images, labels, min_sizes, max_sizes = zip(*data)
    H, W = min_sizes[0], max_sizes[0]
    images = [preprocess(torch.from_numpy(img), min_size=H, max_size=W) for img in images]
    images = torch.stack(images, dim=0)
    
    return images, labels

def resize_image(image: torch.tensor, min_size=600, max_size=1024):
    # Rescaling Images
    C, H, W = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    
    resize = tfs.Resize([int(H*scale), int(W*scale)], antialias=True)
    image = image / 255.
    image = resize(image)
    
    return image*255.

def normalize_tensor_image(image_tensor):
    norm = tfs.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    image = norm(image_tensor)
    return image

def padding(image:torch.tensor, min_size=600, max_size=1024):
    _, H, W = image.size()
    bottom = max_size - H
    right = max_size - W
    
    padder = tfs.Pad([0, 0, right, bottom])
    padded_img = padder(image)
    return padded_img