import os, sys
import numpy as np
import pytorch_lightning as pl
import os
import torch
import torchvision
import torchvision.transforms as transforms
from math import ceil
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from get_cifar import get_data
from helpers_lightning import LightweightLightning

checkpoint_callback = ModelCheckpoint(
    save_top_k=2,           
    monitor="val_epoch_acc",     
    mode="max"              
)

def main(args:dict):

    if args.seed is None:
        args.seed = torch.seed()

    torch.manual_seed(args.seed)
    
    if args.data == 'cifar5m': # For replicating good online learners
        train_dl, val_dl = get_data(five_m = True,
                batch_size = args.batch_size
                )

    elif args.data == 'cifar10': # For traditional CIFAR10 data
        train_dl, val_dl = get_data(custom = args.custom_cifar, five_m = False,
                batch_size = args.batch_size,
                repeat = args.repeat,
                kind = args.kind_custom,
                seed = args.seed
                )

    title = f'{args.experiment}_data_{args.data}_iter_{args.iter}_max_epochs_{args.max_epochs}'
    
    if args.custom_cifar:
        title = f'{args.experiment}_data_{args.data}_{args.kind_custom}_iter_{args.iter}_max_epochs_{args.max_epochs}'
        
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    max_epochs = args.max_epochs

    wandb_logger = WandbLogger(log_model=False, entity='jmryan', name = title, project="emergence_redo", config=args)

    print('---------------------------------------')
    print(title)
    print('---------------------------------------')
    print(f'MAX EPOCHS: {max_epochs}')
    print('---------------------------------------')
    print()

    pl_model = LightweightLightning(args, title)
    trainer = pl.Trainer(accelerator = device, logger = wandb_logger, max_epochs = max_epochs, devices= 'auto', callbacks=[checkpoint_callback])
    trainer.fit(model=pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    wandb.finish()