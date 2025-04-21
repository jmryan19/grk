from argparse import ArgumentParser
from install_packages import *
install_packages()

from Config import Config

import wandb
KEY = '8b81e715f744716c02701d1b0a23c4342e62ad45'
wandb.login(key = KEY)

from grok_train import main as grok_main
from cifar_train import main as cifar_main

    

if __name__ == "__main__":
    for i in range(5):
    # # Grok Baseline
    #     for val in [0.5]:
    #         config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_baseline_no_shuffle', iter = i, data = val, custom_cifar=False, repeat = False, kind_custom = 'top')
    #         grok_main(config)

        # Grok Experiment Unnormed L2 End
        for val in [0.5]:
            config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means', iter = i, data = val, custom_cifar=True, repeat = False, kind_custom = 'unnormed_l2_end')
            grok_main(config)
        
        # Grok Experiment Global Normed L2 End
        for val in [0.5]:
            config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means', iter = i, data = val, custom_cifar=True, repeat = False, kind_custom = 'global_normed_l2_end')
            grok_main(config)
        
        # Grok Experiment Unnormed Cos End
        for val in [0.5]:
            config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means', iter = i, data = val, custom_cifar=True, repeat = False, kind_custom = 'unnormed_cos_end')
            grok_main(config)

        # Grok Experiment Global Normed Cos End
        for val in [0.5]:
            config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means', iter = i, data = val, custom_cifar=True, repeat = False, kind_custom = 'global_normed_cos_end')
            grok_main(config)
                
        
        # # CIFAR Baseline
        # config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'cifar_baseline_final_t', iter = i, save_dist_class_means = True, custom_cifar= False, repeat = False, kind_custom = 'top')
        # cifar_main(config)
        
        # CIFAR Experiment Unnormed L2 End
        config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'cifar_class_means', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = False, kind_custom = 'unnormed_l2_end')
        cifar_main(config)
    
        # CIFAR Experiment Global Normed L2 End
        config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 400, learning_rate = 1e-3, experiment = 'cifar_class_means', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = False, kind_custom = 'global_normed_l2_end')
        cifar_main(config)
    
        # CIFAR Experiment Unnormed Cos End
        config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 350, learning_rate = 1e-3, experiment = 'cifar_class_means', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = False, kind_custom = 'unnormed_cos_end')
        cifar_main(config)

        # CIFAR Experiment Global Normed Cos End
        config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 350, learning_rate = 1e-3, experiment = 'cifar_class_means', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = False, kind_custom = 'global_normed_cos_end')
        cifar_main(config)





        
        # # Grok Experiment Top, Repeat
        # for val in [0.5]:
        #     config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means_top_rpt', iter = i, data = val, custom_cifar=True, repeat = True, kind_custom = 'top')
        #     grok_main(config)
    
        # # Grok Experiment Bottom, Repeat
        # for val in [0.5]:
        #     config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means_bottom_rpt', iter = i, data = val, custom_cifar=True, repeat = True, kind_custom = 'bottom')
        #     grok_main(config)
    
        # # Grok Experiment STD, Repeat
        # for val in [0.5]:
        #     config = Config(max_train_val = 67, training_fraction = val, dim_model = 128, learning_rate = 1e-3, batch_size = 128, max_epochs = 400, plot_inner_products = False, save_dist_class_means = True, experiment='grok_class_means_std_rpt', iter = i, data = val, custom_cifar=True, repeat = True, kind_custom = 'std')
        #     grok_main(config)
                
    
        # # CIFAR Experiment Top, Repeat
        # config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 350, learning_rate = 1e-3, experiment = 'cifar_class_means_top_rpt', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = True, kind_custom = 'top')
        # cifar_main(config)
    
        # # CIFAR Experiment Bottom, Repeat
        # config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 350, learning_rate = 1e-3, experiment = 'cifar_class_means_bottom_rpt', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = True, kind_custom = 'bottom')
        # cifar_main(config)
    
        # # CIFAR Experiment STD, Repeat
        # config = Config(data = 'cifar10', model = 'resnet18', weight_decay = 5e-4, batch_size = 64, max_epochs = 350, learning_rate = 1e-3, experiment = 'cifar_class_means_std_rpt', iter = i, save_dist_class_means = False, custom_cifar= True, repeat = True, kind_custom = 'std')
        # cifar_main(config)

        
