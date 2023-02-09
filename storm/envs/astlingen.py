from pystorms.scenarios import scenario
from .environment_astlingen import env_ast
import os
import yaml
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS
import datetime
from functools import reduce
from itertools import product
from collections import deque

HERE = os.path.dirname(__file__)

class astlingen(scenario):
    r"""Astlingen Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    Objectives are the following:
    1. Minimization of accumulated CSO volume
    2. Minimization of CSO to the creek more than the river
    3. Maximizing flow to the WWTP
    4. Minimizing roughness of control.

    Performance is measured as the following:
    1. *2 for CSO volume (doubled for flow into the creek)
    2. *(-1) for the flow to the WWTP
    3. *(0.01) for the roughness of the control

    """

    def __init__(self, config_file=None, swmm_file=None, global_state=False,initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"config","astlingen.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(HERE,"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        if initialize:
            self.env = env_ast(self.config, ctrl=True)
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}
        self.global_state = global_state # If use global state as input

        # initialize logger
        self.initialize_logger()


    def step(self, actions=None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and 'control_interval' in self.config:
            advance_seconds = self.config['control_interval'] * 60
        # if actions is not None:
        #     actions = self._convert_actions(actions)

        done = self.env.step(actions, advance_seconds = advance_seconds)
        
        # Log the states, targets, and actions
        if log:
            self._logger()

        # Log the performance
        __performance = 0.0
        for ID, attribute, weight in self.config["performance_targets"]:
            __cumvolume = self.env.methods[attribute](ID)
            # Recent volume has been logged
            if len(self.data_log[attribute][ID]) > 1:
                __volume = __cumvolume - self.data_log[attribute][ID][-2]
            else:
                __volume = __cumvolume
            # __weight = self.penalty_weight[ID]
            __performance += __volume * weight

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()
        return done

    def state_full(self):
        __state = np.array([[self.data_log[attr][ID][-1]
            if self.env._isFinished else self.env.methods[attr](ID)
            for ID in self.get_features(typ)] for typ,attr in self.config['global_state']])

        __last = np.array([[self.data_log[attr][ID][-2]
            if attr not in ['depthN','rainfall'] and len(self.data_log[attr][ID]) > 1 else 0
            for ID in self.get_features(typ)] for typ,attr in self.config['global_state']])
        state = (__state - __last).T
        return state

    def state(self, seq = False):
        # Observe from the environment
        if self.global_state:
            state = self.state_full()
            return state
        if self.env._isFinished:
            # if seq:
            #     __state = [list(self.data_log[attribute][ID])[-seq:]
            #     for ID,attribute in self.config["states"]]
            # else:
            __state = [self.data_log[attribute][ID][-1]
            for ID,attribute in self.config["states"]]
        else:
            __state = self.env._state()
            # if seq:
            #     __state = [list(self.data_log[attribute][ID])[-seq:-1] + [__state[idx]]
            #     for idx,(ID,attribute) in enumerate(self.config["states"])]  
        state = []
        for idx,(ID,attribute) in enumerate(self.config["states"]):
            if attribute in ['depthN','rainfall']:
                state.append(__state[idx])
            else:
                # if seq:
                #     if len(self.data_log[attribute][ID]) > seq:
                #         __value = np.diff(self.data_log[attribute][ID][-seq-1:-seq]+__state)
                #     else:
                #         __value = np.diff([0] + __state)
                #     state.append(__value)
                # else:
                if len(self.data_log[attribute][ID]) > 1:
                    __value = __state[idx] - self.data_log[attribute][ID][-2]
                    state.append(__value)
                else:
                    state.append(__state[idx])
        state = np.asarray(state).T if seq else np.asarray(state)
        return state

    def reward(self,norm=False):

        # Calculate the target error in the recent step based on cumulative values
        __reward = 0.0
        __sumnorm = 0.0
        for ID, attribute, weight in self.config["reward"]:
            if self.env._isFinished:
                __cumvolume = self.data_log[attribute][ID][-1]
            else:
                __cumvolume = self.env.methods[attribute](ID)
            # Recent volume has been logged
            if len(self.data_log[attribute][ID]) > 1:
                __volume = __cumvolume - self.data_log[attribute][ID][-2]
            else:
                __volume = __cumvolume

            if attribute == "totalinflow" and ID not in ["Out_to_WWTP","system"]:
                if len(self.data_log[attribute][ID]) > 2:
                    __prevolume = self.data_log[attribute][ID][-2] - self.data_log[attribute][ID][-3]
                elif len(self.data_log[attribute][ID]) == 2:
                    __prevolume = self.data_log[attribute][ID][-2]
                else:
                    __prevolume = 0
                __volume = abs(__volume - __prevolume)
            # __weight = self.penalty_weight[ID]
            if ID == 'system':
                __sumnorm += __volume * weight
            else:
                __reward += __volume * weight
        if norm:
            return - __reward/(__sumnorm + 1e-5)
        else:
            return - __reward

    def reset(self,swmm_file=None,seq=False):
        # clear the data log and reset the environment
        if swmm_file is not None:
            self.config["swmm_input"] = swmm_file
        if not hasattr(self,'env') or swmm_file is not None:
            self.env = env_ast(self.config, ctrl=True)
        else:
            _ = self.env.reset()

        self.initialize_logger()
        state = self.state(seq)
        return state
        
    def initialize_logger(self, config=None,maxlen=None):
        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": deque(maxlen=maxlen),
            "simulation_time": deque(maxlen=maxlen),
            "setting": {}
        }
        config = self.config if config is None else config
        # Data logger for storing _performance & _state data
        for ID, attribute, _ in config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = deque(maxlen=maxlen)
            
        if self.global_state:
            for typ,attribute in config['global_state']:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                for ID in self.get_features(typ):
                    self.data_log[attribute][ID] = deque(maxlen=maxlen)
        else:
            for ID, attribute in config["states"]:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                self.data_log[attribute][ID] = deque(maxlen=maxlen)

        for ID in config["action_space"].keys():
            self.data_log["setting"][ID] = deque(maxlen=maxlen)
        
        for ID, attribute, _ in config["reward"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = deque(maxlen=maxlen)

    def _logger(self):
        super()._logger()


    def get_action_table(self,if_mac):
        action_table = {}
        actions = [len(v) for v in self.config['action_space'].values()]
        site_combs = product(*[range(act) for act in actions])
        for idx,site in enumerate(site_combs):
            if if_mac:
                action_table[site] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
            else:
                action_table[(idx,)] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
        return action_table

    def get_args(self,if_mac = True):
        args = self.config.copy()
        # Rainfall timeseries & events files
        if not os.path.isfile(args['rainfall']['rainfall_timeseries']):
            args['rainfall']['rainfall_timeseries'] = os.path.join(HERE,'config',args['rainfall']['rainfall_timeseries']+'.csv')
        if not os.path.isfile(args['rainfall']['rainfall_events']):
            args['rainfall']['rainfall_events'] = os.path.join(HERE,'config',args['rainfall']['rainfall_events']+'.csv')
        if not os.path.isfile(args['rainfall']['training_events']):
            args['rainfall']['training_events'] = os.path.join(HERE,'config',args['rainfall']['training_events']+'.csv')

        # state shape
        args['state_shape'] = (len(self.get_features('nodes')),len(self.config['global_state'])) if self.global_state else len(args['states'])
        if self.global_state:
            args['edges'] = self.get_edge_list()
        if if_mac:
            # multi-agent controller structure
            args["n_agents"] = len(args['site'])

            # Specify the observe data for each site
            state = [s[0] for s in args['states']]
            args['observ_space'] = [[state.index(o) for o in v['states']]
            for v in args['site'].values()]

            args['action_shape'] = [len(args['action_space'][k]) for k in args['site']]

        else:
            args['n_agents'] = 1

            args['observ_space'] = args['state_shape']  # int value
            
            actions = [len(v) for v in args['action_space'].values()]
            args['action_shape'] = reduce(lambda x,y:x*y,actions)
        args['action_table'] = self.get_action_table(if_mac)
        return args


    # getters
    def get_features(self,kind='nodes'):
        inp = read_inp_file(self.config['swmm_input'])
        labels = {'nodes':NODE_SECTIONS,'links':LINK_SECTIONS}
        features = []
        for label in labels[kind]:
            if label in inp:
                features += list(getattr(inp,label))            
        return features
    
    def get_edge_list(self):
        inp = read_inp_file(self.config['swmm_input'])
        nodes = self.get_features('nodes')
        edges = []
        for label in LINK_SECTIONS:
            if label in inp:
                edges += [(nodes.index(link.FromNode),nodes.index(link.ToNode))
                 for link in getattr(inp,label).values()]
        return np.array(edges)

    # predictive functions
    def save_hotstart(self,hsf_file=None):
        # Save the current state in a .hsf file.
        if hsf_file is None:
            ct = self.env.methods['simulation_time']()
            hsf_file = '%s.hsf'%ct.strftime('%Y-%m-%d-%H-%M')
            hsf_file = os.path.join(os.path.dirname(self.config['swmm_input']),
            self.config['prediction']['hsf_dir'],hsf_file)
        if os.path.exists(os.path.dirname(hsf_file)) == False:
            os.mkdir(os.path.dirname(hsf_file))
        return self.env.save_hotstart(hsf_file)

    def create_eval_file(self,hsf_file=None):
        ct = self.env.methods['simulation_time']()
        inp = read_inp_file(self.config['swmm_input'])

        # Set the simulation time & hsf options
        inp['OPTIONS']['START_DATE'] = inp['OPTIONS']['REPORT_START_DATE'] = ct.date()
        inp['OPTIONS']['START_TIME'] = inp['OPTIONS']['REPORT_START_TIME'] = ct.time()
        inp['OPTIONS']['END_DATE'] = (ct + datetime.timedelta(minutes=self.config['prediction']['eval_horizon'])).date()
        inp['OPTIONS']['END_TIME'] = (ct + datetime.timedelta(minutes=self.config['prediction']['eval_horizon'])).time()
        
        if hsf_file is not None:
            if 'FILES' not in inp:
                inp['FILES'] = FilesSection()
            inp['FILES']['USE HOTSTART'] = hsf_file
        
        # Set the Control Rules
        # inp['CONTROLS'] = Control.create_section()
        # for i in range(self.config['prediction']['control_horizon']//self.config['control_interval']):
        #     time = round(self.config['control_interval']/60*(i+1),2)
        #     conditions = [Control._Condition('IF','SIMULATION','TIME', '<', str(time))]
        #     actions = []
        #     for idx,k in enumerate(self.config['action_space']):
        #         logic = 'THEN' if idx == 0 else 'AND'
        #         kind = self.env.methods['getlinktype'](k)
        #         action = Control._Action(logic,kind,k,'SETTING','=',str('1.0'))
        #         actions.append(action)
        #     inp['CONTROLS'].add_obj(Control('P%s'%(i+1),conditions,actions,priority=5-i))
    

        # Output the eval file
        eval_inp_file = os.path.join(os.path.dirname(self.config['swmm_input']),
                                self.config['prediction']['eval_dir'],
                                self.config['prediction']['suffix']+os.path.basename(self.config['swmm_input']))
        if os.path.exists(os.path.dirname(eval_inp_file)) == False:
            os.mkdir(os.path.dirname(eval_inp_file))
        inp.write_file(eval_inp_file)

        return eval_inp_file

    def get_eval_file(self):
        if self.env._isFinished:
            print('Simulation already finished')
            return None
        else:
            hsf_file = self.save_hotstart()
            eval_file = self.create_eval_file(hsf_file)
            return eval_file

    def get_current_setting(self):
        if len(self.data_log['setting']) > 0 :
            setting = [self.data_log["setting"][ID][-1]
            for ID in self.config["action_space"]]
        else:
            setting = [self.env.methods["setting"](ID)
            for ID in self.config["action_space"]]
        return setting

