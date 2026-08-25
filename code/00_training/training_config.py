
class args_parser:
     def __init__(self):
        # federated arguments (Notation for the arguments followed from paper)
        super(args_parser, self).__init__()
        self.epochs = 10
        self.iteration = 1000000
        # self.num_users = 3 #Manufacturer number fo Federate Learning
        self.frac = 1
        self.local_ep = 1
        self.local_bs = 1
        self.lr = 0.00004
        #self.lr_enc = self.lr/self.num_users
        self.cl_lr = 0.00004
        self.betas =(0.8, 0.99)
        # model arguments
        self.model = 'aesnn'
        self.num_channels = 1
        self.norm = 'None'
        self.dim = 128
        # other arguments
        self.dataset = 'DataSet'
        self.optimizer = 'Adam'
        self.random_seed = 2
        self.training_size = 800
        self.batch_size = 8
        self.loss_record_period = 1000
