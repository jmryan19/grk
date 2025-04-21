from math import ceil
import torch
import numpy as np


def operation_mod_p_data(p: int, eq_token: int, op_token: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    # y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, x).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    labels = (x*y) % p

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels


def get_data(prime: int, training_fraction: float, batch_size: int):
    inputs, labels = operation_mod_p_data(prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

# def get_data_rfm(operation: str, prime: int, training_fraction: float, batch_size: int):
#     inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
#     dataset = torch.utils.data.TensorDataset(inputs, labels)
#     context_len = inputs.shape[1]
#
#     train_size = int(training_fraction * len(dataset))
#     val_size = len(dataset) - train_size
#
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     batch_size = min(batch_size, ceil(len(dataset) / 2))
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#
#     return train_loader, val_loader, context_len
