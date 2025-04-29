class Config():
    def __init__(self, 
                 model = 'resnet18',
                 num_layers = 2,
                 dim_model = 256,
                 num_heads = 4,
                 batch_size = 64,
                 learning_rate = 1e-4,
                 weight_decay = 1,
                 iter = 0,
                 max_epochs = 300,
                 data = 'cifar10',
                 experiment = 'nc_original',
                 save_dist_class_means = True,
                 optim = 'sgd',
                 data_sub = None,
                 seed = None):
        
        self.model = model
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.iter = iter
        self.max_epochs = max_epochs
        self.data = data
        self.experiment = experiment
        self.save_dist_class_means = save_dist_class_means
        self.seed = seed
        self.optim = optim
        self.data_sub = data_sub # '1k', '2class', 'label_change'