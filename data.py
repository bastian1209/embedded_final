import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, datasets, transforms
from augment import MultiviewTransform, base_augment
import os

root = './dataset'

class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

def get_dataset(config, mode='train'):
    root = config.dataset.root
    name = config.dataset.name
    dataset_path = os.path.join(root, name, mode)
    
    if mode.startswith('linear'):
        dataset_path = os.path.join(root, name, 'train')
        
    if mode == 'train':
        transform = MultiviewTransform(config, mode=mode)
    elif mode == 'linear_train':
        transform = base_augment(config, mode='train')
    else:
        transform = base_augment(config, mode=mode)
    
    if name == 'ImageNet_100':
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        
    if name.startswith('cifar'):
        dataset = datasets.CIFAR10(root=root, download=True, train=(mode != 'val'), transform=transform)
    
    return dataset

    
def get_loader(config, dataset, sampler=None):
    if sampler:
        return DataLoader(dataset, batch_size=config.train.batch_size, drop_last=True, num_workers=config.system.num_workers, pin_memory=True, sampler=sampler)
    return DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.system.num_workers, drop_last=True)