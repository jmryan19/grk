import numpy as np
import pytorch_lightning as pl
import os, sys
import torch
from torch import nn
import torchvision
from einops import rearrange, repeat
# sys.path.append(os.getcwd()[:-7])

import wandb
import random

class SanityLightning(pl.LightningModule):
    def __init__(self, config, title):
        super().__init__()
        
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        
        # PyTorch (CPU)
        torch.manual_seed(seed)
            
        # For deterministic behavior (optional, may slow things down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cli_config = config
        self.title = title

        if config.model == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
            model.fc = nn.Identity()
        
        elif config.model == 'mlp':
            layers = []
            layers.append(nn.Linear(32*32*3, 512))
            layers.append(nn.ReLU())
            for i in range(self.cli_config.num_layers):
                layers.append(nn.Linear(512, 512))
                layers.append(nn.ReLU())
            model = nn.Sequential(*layers)

        self.classifier = nn.Linear(512, 10)

        self.model = model
        
        # SETUP Last Layer ACTIVATION STORAGE
        self.train_step_last_layer_activations = []
        self.train_step_labels = []
        self.train_step_idxs = []
        self.train_step_y_hat = []
        
        self.validation_step_last_layer_activations = []
        self.validation_step_labels = []
        self.validation_step_idxs = []
        self.validation_step_y_hat = []

        if config.save_dist_class_means:
            os.makedirs(f'class_means_distance/{title}/train', exist_ok = True)

        
        #SETUP LOSS FUNCTION
        self.loss = nn.CrossEntropyLoss()

    def on_fit_start(self):
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.cli_config.seed)
            torch.cuda.manual_seed_all(self.cli_config.seed)
            torch.use_deterministic_algorithms(True)


    def forward(self, inputs):
        last_layer_activations = self.model(inputs)
        
        outputs = self.classifier(last_layer_activations)

        return outputs, last_layer_activations.detach()
    
    def training_step(self, batch, batch_idx):
        inputs, labels, idxs = batch
        outputs, last_layer_activations = self(inputs)
        
        loss = self.loss(outputs, labels)
        acc = (torch.argmax(outputs, dim=1) == labels).sum() / len(labels)
        y_hats = torch.argmax(outputs, dim = 1)

        _ = [self.train_step_last_layer_activations.append(last_layer_activation) for last_layer_activation in last_layer_activations]
        _ = [self.train_step_labels.append(label) for label in labels]
        _ = [self.train_step_idxs.append(idx) for idx in idxs]
        _ = [self.train_step_y_hat.append(y_hat) for y_hat in y_hats]
        
        self.log("train_epoch_loss", loss, sync_dist=True, on_epoch = False)
        self.log("train_epoch_acc", acc,  sync_dist=True, on_epoch = False)
        # self.log("train_average_predicted_class", torch.argmax(outputs, dim=1).float().mean(), sync_dist=True)
        # self.log("examples_seen", self.global_step * self.cli_config.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, idxs = batch
        outputs, last_layer_activations = self(inputs)
        outputs.detach()

        loss = self.loss(outputs, labels)
        acc = (torch.argmax(outputs, dim=1) == labels).sum() / len(labels)
        y_hats = torch.argmax(outputs, dim = 1)

        _ = [self.validation_step_last_layer_activations.append(last_layer_activation) for last_layer_activation in last_layer_activations]
        _ = [self.validation_step_labels.append(label) for label in labels]
        _ = [self.validation_step_idxs.append(idx) for idx in idxs]
        _ = [self.validation_step_y_hat.append(y_hat) for y_hat in y_hats]
        
        self.log("val_epoch_loss", loss, on_epoch = True, reduce_fx = 'mean', sync_dist=True)
        self.log("val_epoch_acc", acc, on_epoch = True, reduce_fx = 'mean', sync_dist=True)
        # self.log("val_average_predicted_class", torch.argmax(outputs, dim=1).float().mean(), sync_dist=True)
        
        return loss

    def process_activations(self, all_activations, all_labels, all_idxs, all_y_hats, train_val):
        if self.cli_config.save_dist_class_means and self.current_epoch % 5 == 0 and train_val == 'train':
            all_labels = rearrange(all_labels, 'obs -> obs 1') 
            all_y_hats = rearrange(all_y_hats, 'obs -> obs 1') 
            all_idxs = rearrange(all_idxs, 'obs -> obs 1') 
            activations_label_y_hat_idx = torch.hstack((all_activations, all_labels, all_y_hats, all_idxs))

            torch.save(activations_label_y_hat_idx, f'class_means_distance/{self.title}/{train_val}/train_raw_activations_epoch_{self.current_epoch}')

    def on_train_epoch_end(self):
        all_activations = torch.vstack(self.train_step_last_layer_activations)
        all_labels = torch.tensor(self.train_step_labels, device = all_activations.device)
        all_idxs = torch.tensor(self.train_step_idxs, device = all_activations.device)
        all_y_hats = torch.tensor(self.train_step_y_hat, device = all_activations.device)
        
        if self.trainer.num_devices > 1:
            all_activations = rearrange(self.all_gather(all_activations), 'gpus obs dim -> (gpus obs) dim')
            all_labels = rearrange(self.all_gather(all_labels), 'gpus obs -> (gpus obs)')
            all_idxs = rearrange(self.all_gather(all_idxs), 'gpus obs -> (gpus obs)')
            all_y_hats = rearrange(self.all_gather(all_y_hats), 'gpus obs -> (gpus obs)')

        self.train_step_last_layer_activations.clear()
        self.train_step_labels.clear()
        self.train_step_idxs.clear()
        self.train_step_y_hat.clear()

        if self.global_rank == 0:
            self.process_activations(all_activations.detach(), all_labels.detach(), all_idxs.detach(), all_y_hats.detach(), 'train')
    
    def on_validation_epoch_end(self):

        self.validation_step_last_layer_activations.clear()
        self.validation_step_labels.clear()
        self.validation_step_idxs.clear()
        self.validation_step_y_hat.clear()

        # self.process_activations(all_activations, all_labels, all_idxs, all_y_hats, 'val')
    
    def configure_optimizers(self):
        if self.cli_config.optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr = self.cli_config.learning_rate, momentum = 0.9)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.cli_config.max_epochs * 1/3), int(self.cli_config.max_epochs * 2/3)], gamma=0.1)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr= self.cli_config.learning_rate, betas = (0.9, 0.98), weight_decay= self.cli_config.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters=9)
        
        return [optimizer], [lr_scheduler]