import os
import glob
import random
import torch
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd

class CIFAR5M_ImageDataset(torch.utils.data.Dataset):
    def __init__(self, set_path, transform=None):
        """
        PyTorch Dataset for CIFAR-5M images stored as PNG files.

        Args:
            image_paths (list): List of image file paths.
            transform (callable, optional): Transformations to apply to images.
        """
        data = torch.load(set_path, weights_only=False)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.images[idx])

        return image, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(idx, dtype = torch.long)

class CIFAR10_Wrapper(torch.utils.data.Dataset):
    def __init__(self, train = True):
        name  = 'cifar10'
        if not train: name += '_test'
            
        self.data = torchvision.datasets.CIFAR10(name, train= train, transform = transforms.Compose([transforms.ToTensor()]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads an image and its label from the file path.
        """
        image, label = self.data[idx]

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype = torch.long)

class CIFAR10_Custom(torch.utils.data.Dataset):
    def __init__(self, train = True, repeat = False, kind = 'top'):
        name  = 'cifar10'
        if not train: name += '_test'
            
        self.data = torchvision.datasets.CIFAR10(name, train = train, transform = transforms.Compose([transforms.ToTensor()]))
        self.train = train
        self.repeat = repeat

        if train:
            self.indices = torch.load(f'baseline_indexes/{kind}_cifar', weights_only = False).astype(np.int32)
        else:
            self.indices = None

    def __len__(self):
        if self.train:
            if self.repeat:
                return len(self.data)
            else:
                return len(self.indices)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        """
        Loads an image and its label from the file path.
        """
        if self.train:
            if self.repeat:
                if torch.rand(1) <= .25:
                    idx = np.random.choice(self.indices).item()
            else:
                idx = self.indices[idx]

        image, label = self.data[idx]

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype = torch.long)

def get_data(custom = False, five_m = False, batch_size = 64, num_training_sets = 1, num_workers = 4, repeat = False, kind = 'top', seed = None):

    torch.manual_seed(seed)
    
    if five_m: # For replicating good online learners
        all_sets = glob.glob('cifar5m/*.pt')
    
        random.shuffle(all_sets)
    
        train_set = all_sets[0]
        val_set = all_sets[-1]
    
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
        # Create dataset instances
        train_dataset = CIFAR5M_ImageDataset(train_set, transform=transform)
        val_dataset = CIFAR5M_ImageDataset(val_set, transform=transform)
        
    elif custom: # For trying different variations of sets determined by class mean distance
        train_dataset = CIFAR10_Custom(train=True, repeat = repeat, kind = kind)
        val_dataset = CIFAR10_Custom(train=False)
        
    else:
        train_dataset = CIFAR10_Wrapper(train=True)
        val_dataset = CIFAR10_Wrapper(train=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader