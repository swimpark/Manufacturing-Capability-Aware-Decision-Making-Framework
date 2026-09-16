
class args_parser:
     def __init__(self):
        # Training settings
        super(args_parser, self).__init__()
        self.epochs = 10
        self.iteration = 1000000

        self.frac = 1
        self.local_ep = 1
        self.local_bs = 1
        self.lr = 0.00004

        self.cl_lr = 0.00004
        self.betas =(0.8, 0.99)
        # Model settings
        self.model = 'aesnn'
        self.num_channels = 1
        self.norm = 'None'
        self.dim = 128
        # Dataset and logging settings
        self.dataset = 'DataSet'
        self.optimizer = 'Adam'
        self.random_seed = 2
        self.training_size = 800
        self.batch_size = 8
        self.loss_record_period = 1000
