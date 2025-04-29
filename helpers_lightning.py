import numpy as np
import pytorch_lightning as pl
import os, sys
import torch
from torch import nn
import torchvision
from einops import rearrange, repeat
sys.path.append(os.getcwd() + '/grokking/grokking')
sys.path.append(os.getcwd()[:-7] + '/grokking/grokking')
from model import Transformer
import matplotlib.pyplot as plt
import wandb
import random

class LightweightLightning(pl.LightningModule):
    def __init__(self, config, title):
        super().__init__()

        self.cli_config = config
        self.title = title

        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        
        # PyTorch (CPU)
        torch.manual_seed(seed)
        
        # PyTorch (CUDA)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # For deterministic behavior (optional, may slow things down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # SETUP MODEL
        if config.model == 'transformer':
            if config.openai: # Using traditional max_train_val x max_train_val cartesian product space
                num_tokens = config.max_train_val + 2
            else: # Only for replicating emergent properties with repeated examples (without base1000)
                num_tokens = 1_000_000 + 2
            seq_len = 4
            
                
            model = Transformer(
                num_layers= config.num_layers,
                dim_model= config.dim_model,
                num_heads= config.num_heads,
                num_tokens= num_tokens,
                seq_len= seq_len,
                config = config
                )

            self.classifier = nn.Linear(config.dim_model, config.prime)
        
        elif config.model == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
            model.fc = nn.Identity()

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

        # Create Directories for Special Tracking
        if config.plot_inner_products:
            os.makedirs(f'inner_product_plots/{title}', exist_ok = True)
            os.makedirs(f'inner_product_plots/{title}/train', exist_ok = True)
            os.makedirs(f'inner_product_plots/{title}/val', exist_ok = True)
        if config.save_dist_class_means:
            os.makedirs(f'class_means_distance/{title}/train', exist_ok = True)

        
        #SETUP LOSS FUNCTION
        if config.loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()


    def forward(self, inputs):
        last_layer_activations = self.model(inputs)
        
        outputs = self.classifier(last_layer_activations)

        if self.cli_config.model == 'transformer':
            outputs = outputs[-1,:,:]
            last_layer_activations= last_layer_activations[-1,:,:]

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
        
        self.log("train_step_loss", loss, sync_dist=True)
        self.log("train_step_acc", acc, sync_dist=True)
        self.log("train_average_predicted_class", torch.argmax(outputs, dim=1).float().mean(), sync_dist=True)
        self.log("examples_seen", self.global_step * self.cli_config.batch_size, sync_dist=True)

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
        self.log("val_average_predicted_class", torch.argmax(outputs, dim=1).float().mean(), sync_dist=True)
        
        return loss

    def process_activations(self, all_activations, all_labels, all_idxs, all_y_hats, train_val):
        # Order Everything by True Label (Makes it easier for plotting inner products nicely)
        ordered_label_vals, original_label_indices = torch.sort(all_labels)
        all_activations_ord = all_activations[original_label_indices]
        ordered_by_label_idxs = all_idxs[original_label_indices]

        global_mean = all_activations_ord.mean(axis=0)
        
        class_means = []
        class_diffs = []
        Sigma_within = torch.zeros((all_activations_ord.shape[1],all_activations_ord.shape[1]), device= all_activations_ord.device, dtype= all_activations_ord.dtype)
        Sigma_total = torch.zeros((all_activations_ord.shape[1],all_activations_ord.shape[1]), device= all_activations_ord.device, dtype= all_activations_ord.dtype)
        Sigma_between = torch.zeros((all_activations_ord.shape[1],all_activations_ord.shape[1]), device= all_activations_ord.device, dtype= all_activations_ord.dtype)
        class_idxs = []

        for i in range(self.classifier.out_features):
            class_activations = all_activations_ord[ordered_label_vals == i]
            class_mean = class_activations.mean(axis=0) - global_mean
            class_means.append(class_mean)
            class_diff = class_activations - repeat(class_mean, 'u -> n u', n = class_activations.shape[0])
            _ = [class_diffs.append(torch.linalg.vector_norm(diff)) for diff in class_diff]
            
            global_diff = class_activations - repeat(global_mean, 'u -> n u', n = class_activations.shape[0])
            class_ordered_idxs = ordered_by_label_idxs[ordered_label_vals == i]
            _ = [class_idxs.append(idx) for idx in class_ordered_idxs]
            
            Sigma_between += torch.outer((class_mean - global_mean), (class_mean - global_mean)) 
            
            for j in range(len(class_diff)):
                # print(torch.outer(class_diff[i], class_diff[i]).shape)
                Sigma_within += torch.outer(class_diff[j], class_diff[j])
                Sigma_total += torch.outer(global_diff[j], global_diff[j])

        Avg_c = torch.tensor([torch.linalg.vector_norm(class_mean) for class_mean in class_means], device = all_activations_ord.device).mean()
        Std_c = torch.tensor([torch.linalg.vector_norm(class_mean) for class_mean in class_means], device = all_activations_ord.device).std()
        # print(train_val, Avg_c, Std_c)
        self.log(f'{train_val}_class_activations_std_avg', Std_c/Avg_c, on_epoch=True)#, sync_dist=True)

        cos_us = []
        all_angles = []
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                cos_u = torch.inner(class_means[i], class_means[j]) / (torch.linalg.vector_norm(class_means[i]) * torch.linalg.vector_norm(class_means[j]))
                cos_us.append(cos_u)
                all_angles.append(abs(cos_u + (1/ (self.classifier.out_features - 1))))
        
        Avg_angles = torch.tensor(all_angles, device = all_activations_ord.device).mean()
        Std_cos = torch.tensor(cos_us, device = all_activations_ord.device).std()
        self.log(f'{train_val}_class_activations_std_cos', Std_cos, on_epoch = True)#, sync_dist=True)
        
        Sigma_between = Sigma_between / self.classifier.out_features
        Sigma_within = Sigma_within / all_activations.shape[0]
        Sigma_total = Sigma_total / all_activations.shape[0]

        self.log(f'{train_val}_trace_w_trace_b_ratio', torch.trace(Sigma_within) / torch.trace(Sigma_between), on_epoch=True)#, sync_dist=True)

        if self.cli_config.plot_inner_products and self.current_epoch % 50 == 0:
            centered_activations = all_activations_ord - repeat(global_mean, 'u -> n u', n = all_activations_ord.shape[0])
            normed_activations = centered_activations / rearrange(torch.linalg.vector_norm(centered_activations, dim = 1), 'b -> b 1')
            
            inners = normed_activations @ normed_activations.T
        
            plt.imshow(inners.cpu(), cmap='viridis')
            plt.colorbar()
            plt.gca().xaxis.set_visible(False)
            plt.gca().yaxis.set_visible(False)
            plt.title(f'{train_val} Activation Inner Products Epoch {self.current_epoch}')
            plt.savefig(f'inner_product_plots/{self.title}/{train_val}/epoch_{self.current_epoch}', dpi=300, bbox_inches='tight')
            plt.cla()
            plt.clf()
            plt.close()
    
            self.logger.log_image(key=f'{train_val}_inners', images = [inners], caption = [f'Epoch: {self.current_epoch}'])


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
        all_activations = torch.vstack(self.validation_step_last_layer_activations)
        all_labels = torch.tensor(self.validation_step_labels, device = all_activations.device)
        all_idxs = torch.tensor(self.validation_step_idxs, device = all_activations.device)
        all_y_hats = torch.tensor(self.validation_step_y_hat, device = all_activations.device)
        
        if self.trainer.num_devices > 1:
            all_activations = rearrange(self.all_gather(all_activations), 'gpus obs dim -> (gpus obs) dim')
            all_labels = rearrange(self.all_gather(all_labels), 'gpus obs -> (gpus obs)')
            all_idxs = rearrange(self.all_gather(all_idxs), 'gpus obs -> (gpus obs)')
            all_y_hats = rearrange(self.all_gather(all_y_hats), 'gpus obs -> (gpus obs)')

        self.validation_step_last_layer_activations.clear()
        self.validation_step_labels.clear()
        self.validation_step_idxs.clear()
        self.validation_step_y_hat.clear()

        # self.process_activations(all_activations, all_labels, all_idxs, all_y_hats, 'val')
    
    def configure_optimizers(self):
        if self.cli_config.experiment == 'nc_original':
            optimizer = torch.optim.SGD(self.parameters(), lr = self.cli_config.learning_rate, momentum = 0.9, weight_decay = self.cli_config.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.cli_config.max_epochs * 1/3), int(self.cli_config.max_epochs * 2/3)], gamma=0.1)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr= self.cli_config.learning_rate, betas = (0.9, 0.98), weight_decay= self.cli_config.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters=9)
        return [optimizer], [lr_scheduler]