
import os
HERE = os.path.dirname(__file__)

class Arguments:
    def __init__(self,env_args=None,hyp=None):

        ''' Argurments for environment'''
        self.env_args = env_args
        self.env_name = env_args['env_name']
        for k,v in env_args.items():
            setattr(self,k,v)
        
        '''Arguments for agents'''
        self.agent_class = 'VDN'
        self.net_dim = 128
        self.num_layer = 3
        self.if_recurrent = False
        self.seq_len = 3
        self.if_dueling = True
        
        '''Argurments for exploration'''
        self.explore_events = 20
        self.explore_step = 36
        self.total_episodes = 5e3
        self.pre_episodes = 5e3
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.max_capacity = 250000


        '''Argurments for training'''
        self.max_capacity = 250000
        self.batch_size = 256
        self.gamma = 0.97
        self.learning_rate = 1e-5
        self.update_interval = 0.005
        self.repeat_times = 2
        self.loss_function = 'MeanSquaredError'
        self.optimizer = 'Adam'


        '''Arguments for device'''  # TODO
        self.thread_num = 8     # cpu_num
        self.random_seed = 42   # initialize random seed in self.init_before_training()
        self.learner_gpus = 0   # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.eval_gap = 10  # evaluate the agent after the gap
        self.save_gap = 10  # save the agent after the gap
        self.cwd = None # working dir to save the model
        self.if_remove = True # if remove the cwd or keep it

        for k, v in hyp.items():
            setattr(self, k, v)

    def init_before_training(self):
        if self.cwd is None:
            wkdir = os.path.split(HERE)[0]
            self.cwd = os.path.join(wkdir,'model','{0}_{1}'.format(self.env_name, self.agent_class))
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd,ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd,exist_ok=True)
        os.makedirs(os.path.join(self.cwd,'train'),exist_ok=True)
        os.makedirs(os.path.join(self.cwd,'test'),exist_ok=True)
        print(f"| Arguments Create cwd: {self.cwd}")