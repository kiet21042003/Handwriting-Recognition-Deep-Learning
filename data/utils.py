import torch
import torch.nn as nn
import cv2
import torchvision.transforms as tfs
import numpy as np

def get_train_transform(max_size, min_size):
     # assume H < W
    general_augmenter = tfs.Compose([
        tfs.Resize(size=(min_size, max_size)),
        tfs.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
        tfs.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return general_augmenter

def get_val_transform(max_size, min_size):
    val_augmenter = tfs.Compose([
        tfs.Resize(size=(min_size, max_size)),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
    ])
    return val_augmenter

def preprocess(image: torch.tensor, min_size, max_size):
    _, H, W = image.size()
    image = resize_image(image, min_size=min_size, max_size=max_size)
    image = padding(image, min_size=min_size, max_size=max_size)
    image = normalize_tensor_image(image)
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