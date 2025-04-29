import os
import sys
import glob
import random
import torch
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd
from collections import defaultdict

class CIFAR10_Wrapper(torch.utils.data.Dataset):
    def __init__(self, train = True, model = 'resnet18'):
        name  = '../cifar10'
        if not train: name += '_test'
            
        self.data = torchvision.datasets.CIFAR10(name, train= train, transform = transforms.Compose([transforms.ToTensor()]))
        self.model = model
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads an image and its label from the file path.
        """
        image, label = self.data[idx]
        if self.model == 'mlp':
            image = image.flatten()

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype = torch.long)


class MNIST_Wrapper(torch.utils.data.Dataset):
    def __init__(self, train = True, model = 'resnet18'):
        name  = '../mnist'
        if not train: name += '_test'
            
        transform = transforms.Compose([
                                    transforms.Resize((32, 32)),         
                                    transforms.Grayscale(num_output_channels=3),  
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) 
                                    ])
        self.data = torchvision.datasets.MNIST(name, train= train, transform = transform)
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads an image and its label from the file path.
        """
        image, label = self.data[idx]
        if self.model == 'mlp':
            image = image.flatten()

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype = torch.long)


class CIFAR10_Subs(torch.utils.data.Dataset):
    def __init__(self, train = True, model = 'resnet18', sub = '1k'):
        name  = '../cifar10'
        if not train: name += '_test'
            
        
        self.data = torchvision.datasets.CIFAR10(name, train= train, transform = transforms.Compose([transforms.ToTensor()]))
        self.indices = []
        if sub == '1k':
            dic = defaultdict(list)
            for i, (_, lab) in enumerate(self.data):
                if len(dic[lab]) == 100:
                    continue
                else:
                    dic[lab].append(i)
            self.indices = [item for sublist in dic.values() for item in sublist]
            
        self.sub = sub
        self.model = model
    def __len__(self):
        if self.sub == '1k':
            return len(self.indices)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        """
        Loads an image and its label from the file path.
        """
        if self.sub == '1k':
            idx = self.indices[idx]

        image, label = self.data[idx]
        
        if self.sub == '2class':
            zeros = set([0,1,8,9])
            if label in zeros:
                label = 0
            else:
                label = 1

        if self.sub == 'label_change':
            if idx == 9749:
                label = 5
            elif idx == 28898:
                label = 6
            elif idx == 49412:
                label = 9
            elif idx == 9282:
                label = 3
            elif idx == 36567:
                label = 5
            elif idx == 4740:
                label = 0
            elif idx == 39328:
                label = 9
            elif idx == 16927:
                label = 5
            elif idx == 44496:
                label = 1
            elif idx == 2688:
                label = 2

        # if self.model == 'mlp':
        #     image = image.flatten()

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype = torch.long)


def get_data(data = 'cifar10', batch_size = 64, num_workers = 4, seed = None, model = 'resnet18', sub = None):

    torch.manual_seed(seed)
    
    if data == 'cifar10':
        if sub is None:
            train_dataset = CIFAR10_Wrapper(train=True, model = model)
            val_dataset = CIFAR10_Wrapper(train=False, model = model)
        else:
            train_dataset = CIFAR10_Subs(train=True, model = model, sub = sub)
            val_dataset = CIFAR10_Subs(train=False, model = model, sub = sub)
    elif data == 'mnist':
        train_dataset = MNIST_Wrapper(train=True, model = model)
        val_dataset = MNIST_Wrapper(train=False, model = model)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader