import torch.nn as nn
import torchvision 
from torchvision import transforms
import cv2
from PIL import ImageFilter
import numpy as np
import random


class MultiviewTransform:
    def __init__(self, config, mode='train', num_view=2):
        self.config = config
        self.num_view = num_view
        self.transform = base_augment(config, mode=mode)
    
    def __call__(self, sample):
        views = []
        for _ in range(self.num_view):
            view = self.transform(sample)
            views.append(view)
        
        return views
    

def base_augment(config, mode='train'):
    img_size = config.dataset.img_size[0]
    crop = transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.))
    flip = transforms.RandomHorizontalFlip()
    color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) # simclr -> 0.4, 0.4, 0.4, 0.2
    gray_scale = transforms.RandomGrayscale(0.2)
    gaussian_blur = transforms.RandomApply([GaussianBlur()], p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transforms_list = np.array([crop, color_jitter, gray_scale, gaussian_blur, flip, to_tensor, normalize])

    if mode == 'train':
        augment_mask = np.array([True, True, True, True, True, True, True])
    elif mode == 'linear':
        augment_mask = np.array([True, False, False, False, True, True, True])
    elif mode == 'val':
        augment_mask = np.array([True, False, False, False, False, True, True])
    else:
        raise NotImplementedError
    
    transform = transforms.Compose(transforms_list[augment_mask])
    
    return transform
    
    
class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return x
        