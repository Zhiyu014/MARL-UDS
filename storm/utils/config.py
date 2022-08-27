from .logger import Trainlogger,Testlogger
from agent import VDN,DQN,IQL
import os
HERE = os.path.dirname(__file__)

class Arguments:
    def __init__(self,env_args=None,hyp=None):

        ''' Argurments for environment'''
        self.env_args = env_args
        if env_args is not None:
            self.env_name = env_args['env_name']
            for k,v in env_args.items():
                setattr(self,k,v)
        
        '''Arguments for agents'''
        self.agent_class = 'VDN'
        self.if_mac = True
        self.net_dim = 128
        self.num_layer = 3
        self.if_recurrent = False
        self.seq_len = 3
        self.if_dueling = True
        
        '''Argurments for exploration'''
        self.explore_events = 20
        self.explore_step = 36
        self.total_episodes = 5e3
        self.pre_episodes = 100
        self.ini_episodes = 0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.epsilon = 1


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
        self.eval_events = 1
        self.eval_gap = 10  # evaluate the agent after the gap
        self.save_gap = 100  # save the agent after the gap
        self.cwd = None # working dir to save the model
        self.if_remove = True # if remove the cwd or keep it
        self.if_load = True # if load the current model in the cwd
        self.replace_rain = False # if replace the rainfall events


        '''Arguments for test'''
        self.test_events = 8
        self.replace_rain = False # if replace the rainfall events
        self.test_agents = ['VDN']
        self.if_predict = False
        self.cwd = None

        '''TODO: Arguments for online search'''
        self.algorithm = 'GA'
        self.pop_size = 32
        self.sampling = ('Initialize',0.4)
        self.crossover = ('SBX',1.0,3.0)
        self.mutation = ('PM',1.0,3.0)
        self.termination = ('time','00:05:00')
        
        if hyp is not None:
            for k, v in hyp.items():
                setattr(self, k, v)



    def init_before_training(self,load=None):

        self.epsilon = max(self.epsilon_decay**max(self.ini_episodes-self.pre_episodes,0),self.epsilon_min)
        self.agent_class = eval(self.agent_class)

        if self.cwd is None:
            wkdir = os.path.split(HERE)[0]
            self.cwd = os.path.join(wkdir,'model','{0}_{1}'.format(self.env_name, self.agent_class))
        else:
            self.cwd = os.path.abspath(self.cwd)
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd,ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd,exist_ok=True)
        os.makedirs(os.path.join(self.cwd,'train'),exist_ok=True)
        os.makedirs(os.path.join(self.cwd,'eval'),exist_ok=True)
        os.makedirs(os.path.join(self.cwd,'reward'),exist_ok=True)
        print(f"| Arguments Create cwd: {self.cwd}")

        load = self.if_load if load is None else load
        log = Trainlogger(self.cwd,load)
        return log

    def init_before_testing(self):
        self.agent_class = eval(self.agent_class)
        self.if_load = True
        self.if_remove = False
        if self.cwd is None:
            wkdir = os.path.split(HERE)[0]
            self.cwd = os.path.join(wkdir,'model','{0}_{1}'.format(self.env_name, self.agent_class))
        else:
            self.cwd = os.path.abspath(self.cwd)
        print(f"| Arguments Keep cwd: {self.cwd}")

    def init_test(self,load=None):
        if self.cwd is None:
            wkdir = os.path.split(HERE)[0]
            self.cwd = os.path.join(wkdir,'result',self.env_name)
        else:
            self.cwd = os.path.abspath(self.cwd)

        load = self.if_load if load is None else load
        log = Testlogger(self.cwd,load)


        if not load and self.if_remove:
            import shutil
            shutil.rmtree(self.cwd,ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd,exist_ok=True)
        print(f"| Arguments Create cwd: {self.cwd}")

        return log