import wandb
KEY = '8b81e715f744716c02701d1b0a23c4342e62ad45'
wandb.login(key = KEY)

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

from sanity_get_data import get_data
from sanity_lightning import SanityLightning

# sys.path.append(os.getcwd()[:-7])
# from helpers_lightning import LightweightLightning

checkpoint_callback = ModelCheckpoint(
    save_top_k=2,           
    monitor="val_epoch_acc",     
    mode="max"              
)



def main(args:dict):

    if args.seed is None:
        args.seed = torch.seed()

    torch.manual_seed(args.seed)

    train_dl, val_dl = get_data(data = args.data,
                                batch_size = args.batch_size,
                                seed = args.seed,
                                model = args.model,
                                sub = args.data_sub
                                )


    if args.data_sub is None:
        title = f'{args.experiment}_data_{args.data}_{args.model}_seed_{args.seed}_iter_{args.iter}_max_epochs_{args.max_epochs}'
    else:
        title = f'{args.experiment}_data_{args.data}_{args.data_sub}_{args.model}_seed_{args.seed}_iter_{args.iter}_max_epochs_{args.max_epochs}'
    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        return
    max_epochs = args.max_epochs

    wandb_logger = WandbLogger(log_model=False, entity='jmryan', name = title, project="emergence_redo", config=args)

    print('---------------------------------------')
    print(title)
    print('---------------------------------------')
    print(f'MAX EPOCHS: {max_epochs}')
    print('---------------------------------------')
    print()

    pl_model = SanityLightning(args, title)
    trainer = pl.Trainer(accelerator = device, logger = wandb_logger, max_epochs = max_epochs, devices= 'auto', callbacks=[checkpoint_callback])
    trainer.fit(model=pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    wandb.finish()