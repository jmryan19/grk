import os, sys
import numpy as np
import pytorch_lightning as pl
import os
import torch
from math import ceil
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from get_grok import get_data
from helpers_lightning import LightweightLightning


checkpoint_callback = ModelCheckpoint(
    save_top_k=2,           
    monitor="val_epoch_acc",     
    mode="max"              
)

def main(args:dict):

    args.batch_size = min(int(int((args.max_train_val ** 2) * args.training_fraction) / 2), args.batch_size )

    if args.seed is None:
        args.seed = torch.seed()

    torch.manual_seed(args.seed)
    
    train_dl, val_dl, context_len = get_data(
            args.prime,
            args.data_budget,
            args.train_budget,
            args.batch_size,
            args.base,
            args.openai,
            args.training_fraction,
            args.max_train_val,
            args.custom_cifar, 
            repeat = args.repeat, 
            kind = args.kind_custom, 
            seed = args.seed
            )
    
    torch.manual_seed(args.seed)   
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    DB_str = str(args.data_budget)[:-6] + 'M'
    TB_str = str(args.train_budget)[:-6] + 'M'

    max_epochs = args.max_epochs
    
    if not args.openai: # For replicating emergence with repeated examples
        title = f'official_{args.model}_{DB_str}_{TB_str}_{args.iter}'

    elif args.custom_cifar: # For trying different variations of sets determined by class mean distance
        title = f'{args.experiment}_data_{args.data}_{args.kind_custom}_train_fraction_{args.training_fraction}_iter_{args.iter}_max_epochs_{args.max_epochs}'
    
    else: # 'Normal' runs
        title = f'{args.experiment}_data_{args.data}_train_fraction_{args.training_fraction}_iter_{args.iter}_max_epochs_{args.max_epochs}'
    
    wandb_logger = WandbLogger(log_model="all", entity='jmryan', name = title, project="emergence_redo", config=args)
    print(wandb_logger.experiment.config)

    print('---------------------------------------')
    print(title)
    print('---------------------------------------')
    print(f'MAX EPOCHS: {max_epochs}')
    print('---------------------------------------')
    print()


    pl_model = LightweightLightning(args, title = title)
    trainer = pl.Trainer(accelerator = device, logger = wandb_logger, max_epochs = max_epochs, devices= 'auto', callbacks=[checkpoint_callback])
    trainer.fit(model=pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    wandb.finish()