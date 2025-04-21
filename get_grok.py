import torch
import numpy as np
from math import ceil
import pandas as pd

FULL = torch.arange(1,1_000)
torch.manual_seed(42)

def tokenize(n): # For replicating emergence with repeated examples
    base1000_parts = []
    if n == 0:
        base1000_parts.append(0)
    while n > 0:
        base1000_parts.append((n % 1000))
        n //= 1000
    while len(base1000_parts) < 3:
        base1000_parts = base1000_parts + [1000]
    
    return base1000_parts[::-1]

def create_TB(size): # For replicating emergence with repeated examples
    return np.random.choice(FULL, size = int(np.sqrt(size)) + 1, replace = False)

def create_DB(size): # For replicating emergence with repeated examples
    if size == 'population':
        return None #TODO
    else:
        return np.random.choice(FULL, size= int(np.sqrt(size)) + 1, replace = False)

def create_openai_data(p: int, eq_token: int, op_token: int, size = 0):
    x = torch.arange(0, size)
    x,y = torch.cartesian_prod(x,x).T

    z = (x*y)%p
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    inputs = torch.stack([x, op, y, eq], dim=1)
    labels = z

    return inputs, labels

class ToyTest(torch.utils.data.Dataset): # For testing that data is tracked well through training and raw activations saving
    def __init__(self, size=67):
        self.size = torch.arange(size)

    def __len__(self):
        return len(self.size)

    def __getitem__(self, idx):
        value = self.size[idx]  # all outputs are just the index
        input_ = value
        label = value
        return torch.tensor([input_, input_, input_, input_]), torch.tensor(label, dtype=torch.long), torch.tensor(value, dtype=torch.long)

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        return self.inputs[idx], self.labels[idx], torch.tensor(idx, dtype=torch.long)

class DatasetCustomWrapper(torch.utils.data.Dataset):  # For trying different variations of sets determined by class mean distance
    def __init__(self, train_dataset, repeat = False, kind = 'top'):
        self.data = train_dataset
        self.repeat = repeat

        orig_indices = set((torch.load(f'baseline_indexes/{kind}_grok', weights_only=False).astype(np.int32)).tolist())
        self.indices = []
        for i in range(len(self.data)):
            if self.data[i][2].item() in orig_indices:
                self.indices.append(i)
                
    def __len__(self):
        if self.repeat:
            return len(self.data)
        else:
            return len(self.indices)

    def __getitem__(self,idx):
        # print(len(self.indices), idx)
        if self.repeat:
            if torch.rand(1) <= .25:
                idx = np.random.choice(self.indices).item()
        else:
            idx = self.indices[idx]
        # print(len(self.data), idx)
        inputs, label, idx = self.data[idx]

        return inputs, label, idx


class ModTrainDataset(torch.utils.data.Dataset): # For replicating emergence with repeated examples
    def __init__(self, DB_size: int, p: int, eq_token: int, op_token: int, base: bool):
        
        self.DB = torch.tensor(create_DB(DB_size))
        self.cartesian = torch.cartesian_prod(self.DB, self.DB)
        self.p = p
        self.eq_token = eq_token
        self.op_token = op_token
        self.base = base

    def __len__(self):
        return len(self.cartesian)

    def __getitem__(self,idx):
        x,y = self.cartesian[idx]
        z = (x*y)%self.p

        if self.base:
            x_tokenized = torch.tensor(tokenize(x.item()))
            y_tokenized = torch.tensor(tokenize(y.item()))
        else:
            x_tokenized = x
            y_tokenized = y

        input = torch.hstack([x_tokenized, torch.tensor(self.op_token), y_tokenized, torch.tensor(self.eq_token)])
        label = z
        return input, label

class ModValDataset(torch.utils.data.Dataset): # For replicating emergence with repeated examples
    def __init__(self, val_len: int, p: int, eq_token: int, op_token: int, base: bool):
        self.p = p
        self.val_len = val_len
        self.eq_token = eq_token
        self.op_token = op_token
        self.base = base

    def __len__(self):
        return self.val_len

    def __getitem__(self,idx):
        x = int(np.random.choice(FULL))
        y = int(np.random.choice(FULL))
            
        z = (x*y)%self.p

        if self.base:
            x_tokenized = torch.tensor(tokenize(x))
            y_tokenized = torch.tensor(tokenize(y))
        else:
            x_tokenized = torch.tensor(x)
            y_tokenized = torch.tensor(y)

        input = torch.hstack([x_tokenized, torch.tensor(self.op_token), y_tokenized, torch.tensor(self.eq_token)])
        label = z
        return input, label

def get_data(prime: int, data_budget: int, training_budget: int, batch_size: int, base=False, openai=False, training_fraction = 0, max_train_val = None, custom=False, repeat = False, kind = 'top', toy_test = False, seed=None):
    torch.manual_seed(42)

    if base: # For replicating emergence with repeated examples
        eq_token = 1001
        op_token = 1002
        context_len = 8
    else: # For replicating emergence with repeated examples (without base1000)
        eq_token = 1_000_000
        op_token = 1_000_001
        context_len = 4

    if openai:
        eq_token = prime 
        op_token = prime + 1
        context_len = 4

        if max_train_val:
            inputs, labels = create_openai_data(prime, max_train_val, max_train_val + 1, size = max_train_val)
        else:
            inputs, labels = create_openai_data(prime, eq_token, op_token, size= prime)
        
        dataset = DatasetWrapper(inputs, labels)
        if toy_test:
            dataset = ToyTest()
    
        train_size = int(training_fraction * len(dataset))
        val_size = len(dataset) - train_size
    
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        if custom:
            train_dataset = DatasetCustomWrapper(train_dataset, repeat = repeat, kind = kind)
        
    else: # For replicating emergence with repeated examples
        train_dataset = ModTrainDataset(data_budget, prime, eq_token, op_token, base)
        val_dataset = ModValDataset(10_000, prime, eq_token, op_token, base)
    
    batch_size = min(batch_size, ceil(len(train_dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, context_len