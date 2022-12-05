# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:47:01 2022

@author: MOMO
"""
from .environment_RedChicoSur import env_RedChicoSur
# from pystorms.networks import load_network
# from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import perf_metrics
import numpy as np
import yaml
import os
from functools import reduce
from itertools import product
import datetime
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control

HERE = os.path.dirname(__file__)

class RedChicoSur(scenario):
    r"""RedChicoSur Scenario

    Stormwater network in Bogotá, Colombia.

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
    1. Minimization of the flooding
    2. Minimization of node depths

    Performance is measured as the following:
    1. *2 for the flooding
    2. *1 for the node depths

    """
    
    def __init__(self, config_file = None, swmm_file = None, initialize = True):
        # Network configuration
        config_file = os.path.join(HERE,"config","RedChicoSur.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(HERE,"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        if initialize:
            self.env = env_RedChicoSur(self.config, ctrl=True)
            self.node_fullDepth = {node: self.env.methods['getnodefulldepth'](node)
            for node,attribute in self.config['states'] if attribute == 'depthN'}

        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}

        # initialize logger
        self.initialize_logger()

    def step(self, actions=None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and 'control_interval' in self.config:
            advance_seconds = self.config['control_interval'] * 60
        
        done = self.env.step(actions, advance_seconds = advance_seconds)
        
        # Log the states, targets, and actions
        if log:
            self._logger()
        
        # Calculate the target error in the recent step based on cumulative values
        __performance = []
        for info in list(self.config['sites'].values()) + [self.config]:
            __perf = 0
            for ID, attribute, weight in info["performance_targets"]:
                __value = self.env.methods[attribute](ID)
                if attribute in ['depthN']:
                    __perf += __value/self.node_fullDepth[ID] * weight
                # Recent volume has been logged
                else:
                    if attribute.startswith('cum') and len(self.data_log[attribute][ID]) > 1:
                        __value -= self.data_log[attribute][ID][-2]
                    __perf += __value * weight
            __performance.append(__perf)


        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()
        return done

    def state(self):
        # Observe from the environment
        if self.env._isFinished:
            __state = [self.data_log[attribute][ID][-1]
             for ID,attribute in self.config["states"]]
        else:
            __state = self.env._state()

        state = []
        for idx,(ID,attribute) in enumerate(self.config["states"]):
            # Normalize the node depths
            if attribute in ['depthN']:
                state.append(__state[idx]/self.node_fullDepth[ID])
            else:
                # Calculate the recent volume based on cumulative value
                # log must be True
                # Recent volume has been logged
                if attribute.startswith('cum') and len(self.data_log[attribute][ID]) > 1:
                    __value = __state[idx] - self.data_log[attribute][ID][-2]
                    state.append(__value)
                else:
                    state.append(__state[idx])
            
        state = np.asarray(state)
        return state

    def performance(self,metric='recent'):
        # Return the recent target value
        perfs = np.array(self.data_log["performance_measure"]).T
        return [perf_metrics(perf,metric) for perf in perfs]
        # return perf_metrics(self.data_log['performance_measure'],metric)

    def reward(self, global_reward = False, weights = None):
        perfs = self.performance('recent')
        if global_reward:
            rewards = perfs[-1]
        else:
            if weights is None:
                alpha,beta,gamma = self.config['reward']['self'],self.config['reward']['neighbor'],self.config['reward']['system']
            else:
                alpha,beta,gamma = weights
            rewards = alpha*np.array(perfs[:-1]) +\
                np.array([beta * sum([perfs[list(self.config['sites']).index(site)]
                for site in info['communicator']])
            for info in self.config['sites'].values()]) +\
                gamma * perfs[-1]
        return - rewards


    def reset(self,swmm_file=None):
        # clear the data log and reset the environment
        if swmm_file is not None:
            self.config["swmm_input"] = swmm_file
        if not hasattr(self,'env') or swmm_file is not None:
            self.env = env_RedChicoSur(self.config, ctrl=True)
            self.node_fullDepth = {node: self.env.methods['getnodefulldepth'](node)
            for node,attribute in self.config["states"] if attribute == 'depthN'}
        else:
            _ = self.env.reset()

        state = self.state()
        self.initialize_logger()
        return state

    def initialize_logger(self):
        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "setting": {}
        }
        # Data logger for storing _performance & _state data
        for site,info in self.config["sites"].items():
            for ID, attribute, _ in info["performance_targets"]:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                self.data_log[attribute][ID] = []

            for ID, attribute in info["states"]:
                if attribute not in self.data_log.keys():
                    self.data_log[attribute] = {}
                self.data_log[attribute][ID] = []
            
            self.data_log["setting"][site] = []

        for ID, attribute in self.config["states"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []

        for ID, attribute,_ in self.config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []

    def _logger(self):
        super()._logger()

            
    def get_args(self,if_mac = True, if_comm = False):
        # Set the environment arguments
        config = self.config
        args = config.copy()
        # Designed rainfall params
        # args['rainfall_parameters'] = os.path.join(HERE,'config',config['rainfall_parameters']+'.yaml')
        # Control interval for step_advance
        args['state_shape'] = len(args['states'])

        if if_mac:
            # multi-agent controller structure
            args['n_agents'] = len(config['sites'])

            # Specify the observe data for each site
            state = [s[0] for s in args['states']]
            args['observ_space'] = [[state.index(o[0]) for o in info['states']]
                                     for info in config['sites'].values()]
            if if_comm:
                # TODO: Communication
                args['communicate_space'] = [[list(config['sites']).index(site)
                 for site in info['communicator']]
                for info in config['sites'].values()]

            # Calculate the action nums for each site
            args['action_shape'] = [len(info['action_space'])
                                     for info in config['sites'].values()]
        else:
            # deprecated: single-agent
            args['n_agents'] = 1

            args['observ_space'] = args['state_shape']

            # multiply nums of pumps in each site
            actions = [len(info['action_space'])
            for info in config['sites'].values()]
            args['action_shape'] = reduce(lambda x,y:x*y,actions)
        return args

    def get_current_setting(self):
        if len(self.data_log['setting']) > 0 :
            setting = [self.data_log["setting"][ID][-1]
            for ID in self.config["action_space"]]
        else:
            setting = [self.env.methods["setting"](ID)
            for ID in self.config["action_space"]]
        return setting
