import sys
import os
sys.path.append(os.getcwd()[:-7])

from install_packages import *
install_packages()

from sanity_config import Config
import wandb
KEY = '8b81e715f744716c02701d1b0a23c4342e62ad45'
wandb.login(key = KEY)

from sanity_train import main as sanity_main

if __name__ == "__main__":
    for seed in [713]:
        i = 0
        # config = Config(data = 'cifar10', weight_decay = 5e-4, batch_size = 64, max_epochs = 30, learning_rate = 1e-3, experiment = 'test_sanity_no_shuffle', iter = 0, save_dist_class_means = True, seed = seed, model = 'resnet18')
        # sanity_main(config)


        # config = Config(data = 'mnist', weight_decay = 5e-4, batch_size = 64, max_epochs = 30, learning_rate = 1e-3, experiment = 'test_sanity_no_shuffle', iter = 0, save_dist_class_means = True,  seed = seed, model = 'resnet18')
        # sanity_main(config)


        for j in range(3, 7):
            if j > 3:
                config = Config(data = 'cifar10', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = f'low_complexity_{j}', iter = i, save_dist_class_means = True, seed = seed, model = 'mlp', num_layers = j)
                sanity_main(config)
    
            config = Config(data = 'mnist', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = f'low_complexity_{j}', iter = i, save_dist_class_means = True,  seed = seed, model = 'mlp', num_layers = j)
            sanity_main(config)

        # config = Config(data = 'cifar10', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'sanity_no_shuffle', iter = i, save_dist_class_means = True, seed = seed, model = 'resnet18', data_sub = '1k')
        # sanity_main(config)

        # config = Config(data = 'cifar10', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'sanity_no_shuffle', iter = i, save_dist_class_means = True, seed = seed, model = 'resnet18', data_sub = '2class')
        # sanity_main(config)

        # config = Config(data = 'cifar10', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'sanity_no_shuffle', iter = i, save_dist_class_means = True, seed = seed, model = 'resnet18', data_sub = 'label_change')
        # sanity_main(config)