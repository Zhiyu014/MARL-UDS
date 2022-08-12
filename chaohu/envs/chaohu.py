# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:47:01 2022

@author: MOMO
"""
from env.environment_chaohu import env_chaohu
from env.utilities import eval_closing_reward,eval_control
# from pystorms.networks import load_network
# from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import perf_metrics
import numpy as np
import yaml
import os
from functools import reduce
from itertools import product

HERE = os.path.dirname(__file__)

class chaohu(scenario):
    r"""Chaohu Scenario

    Combined sewer network in eastern China.

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
    2. Minimization of CSO to the river
    3(TODO). Minimization of the energy consumption of pumps
    4(TODO). Minimization of the number of startups of pumps

    Performance is measured as the following:
    1. *2 for the flooding
    2. *1 for the CSO
    3(TODO). *10 for the energy consumption

    """
    
    def __init__(self, config_file = None, swmm_file = None):
        # Network configuration
        config_file = os.path.join(HERE,"config","chaohu.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(HERE,"network",self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        self.env = env_chaohu(self.config, ctrl=True)
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}

        self.node_properties = {ID: {'initDepth':self.env.methods['getnodeinitdepth'](ID),
        'fullDepth':self.env.methods['getnodefulldepth'](ID)}
        for ID, attribute in self.config["states"] if attribute in ['depthN']}

        # initialize logger
        self.initialize_logger()

    def step(self, actions=None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions, advance_seconds = advance_seconds)
        
        # Log the states, targets, and actions
        if log:
            self._logger(actions)
        
        # Calculate the target error in the recent step based on cumulative values
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
                state.append(__state[idx]/self.node_properties[ID]['fullDepth'])
            else:
                # Calculate the recent volume based on cumulative value
                # log must be True
                # Recent volume has been logged
                if len(self.data_log[attribute][ID]) > 1:
                    __value = __state[idx] - self.data_log[attribute][ID][-2]
                    state.append(__value)
                else:
                    state.append(__state[idx])
            
        state = np.asarray(state)
        return state

    def performance(self,metric='recent'):
        # Return the recent target value
        return perf_metrics(self.data_log["performance_measure"],metric)


    def reward(self, done = False, baseline = None):
        '''
        Get the reward step-wise reward value
        Reward shaping:
        1. Operational reward: ±2
            1) If raining:  sum[(depth - ini_depth)/ini_depth] - sum[energy] * 0.2
            2) Not raining: sum[(ini_depth - depth)/ini_depth] - sum[energy] * 0.2
            3) If flooding in storage: -5
        2. Closing reward:  ±5 (gamma=0.95 n_steps=36 reward=2/(0.95**36)=4.16 --> 5)
            1) - 8 * (sum[flooding] + sum[overflow])/(sum[outflow] + sum[flooding] + final_storage) + 5
                cannot reach ±5 at all times so multiply -20 plus 15
            OR 2) 10 * (HC[fl&CSO] - fl&CSO)/HC[fl&SCO]
                cannot reach ±5 at all times so multiply 10
            OR 3) if fl&CSO < HC[fl&CSO] reward += 5 else reward -= 5
        '''
        if self.env._isFinished:
            __reward = [self.data_log[attribute][ID][-1]
             for ID,attribute in self.config["reward"]]
        else:
            __reward = [self.env.methods[attribute](ID)
             for ID,attribute in self.config["reward"]]

        reward = dict()
        for idx,(ID,attribute) in enumerate(self.config["reward"]):
            if attribute not in reward.keys():
                reward[attribute] = dict()
            if attribute == 'depthN':
                reward[attribute][ID] = __reward[idx]/self.node_properties[ID]['initDepth']
            else:
                if len(self.data_log[attribute][ID]) > 1:
                    reward[attribute][ID] = __reward[idx] - self.data_log[attribute][ID][-2]
                else:
                    reward[attribute][ID] = __reward[idx]

        value = 0
        if sum(reward['cumprecip'].values()) > 0:
            value += sum([v-1 for v in reward['depthN'].values()])
        else:
            value += sum([1-v for v in reward['depthN'].values()])
        value -= sum(reward['pumpenergy'].values()) * 0.2

        if sum(reward["cumflooding"].values()) > 0:
            value -= 2

        if done and baseline is not None:
            total_flood_cso = self.performance('cumulative')
            # value += 10 * (1 - total_flood_cso/baseline)
            if total_flood_cso > baseline:
                value += -5 - 10 * (total_flood_cso/baseline - 1)
            else:
                value += 5 + 10 * (1 - total_flood_cso/baseline)
        return value

    def reset(self,swmm_file=None):
        # clear the data log and reset the environment
        if swmm_file is None:
            _ = self.env.reset()
            state = self.state()
        else:
            # change the swmm inp file
            self.config["swmm_input"] = swmm_file
            self.env = env_chaohu(self.config, ctrl=True)
            state = self.state()
        self.initialize_logger()
        return state

    def initialize_logger(self):
        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": []
        }
        # Data logger for storing _performance & _state data
        for ID, attribute, _ in self.config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []
            
        for ID, attribute in self.config["states"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []
        
        for ID, attribute in self.config["reward"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []
        
        # Data logger for storing action in each step
        self.action_log = []

    
    def _logger(self,actions):
        super()._logger()
        # Log the actions
        if actions is None:
            self.action_log.append([self.env.methods['getlinksetting'](ele)
                                        for ele in self.config['action_space']])
        else:
            self.action_log.append(actions)

    def get_action_table(self,if_mac):
        actions = [[len(a)+1 for a in v['action_space']]
                    for v in self.config['site'].values()]
        action_table = {}
        if if_mac:
            action_combs = [list(product(*[range(i) for i in num_action]))
                                for num_action in actions]
            site_combs = product(*[range(len(action_comb)) for action_comb in action_combs])

            for site in site_combs:
                action = []
                for idx,(num,num_action) in enumerate(zip(site,action_combs)):
                    acts = num_action[num]
                    for i in range(len(acts)):
                        action += [1]*acts[i] + [0]*(actions[idx][i]-1-acts[i])
                action_table[site] = action
        else:
            action_combs = [list(product(*[range(i) for i in num_action]))
                             for num_action in actions]
            all_combs = list(product(*action_combs))
            for i,comb in enumerate(all_combs):
                action = []
                for com,acts in zip(comb,actions):
                    for c,a in zip(com,acts):
                        action += [1]*c + [0]*(a-1-c)
                action_table[i] = action
        return action_table
            
    def get_args(self,if_mac = True):
        # Set the environment arguments
        config = self.config
        args = {}
        args['env_name'] = config['env_name']
        args['swmm_input'] = config["swmm_input"]
        # Designed rainfall params
        args['rainpara'] = os.path.join(HERE,'config',config['rainpara']+'.yaml')
        # Control interval for step_advance
        args['control_interval'] = config['control_interval']
        args['state_shape'] = len(config['states'])
        if if_mac:
            # multi-agent controller structure
            args['n_agents'] = len(config['site'])

            # Specify the observe data for each site
            state = [s[0] for s in config['states']]
            args['observ_space'] = [[state.index(o) for o in v['states']]
                                     for v in config['site'].values()]

            # Calculate the action nums for each site
            actions = [[len(a)+1 for a in v['action_space']]
                        for v in config['site'].values()]
            args['action_space'] = [reduce(lambda x,y:x*y,action)
                                     for action in actions]
        else:
            # single-agent
            args['n_agents'] = 1

            # multiply nums of pumps in each site
            actions = [[len(a)+1 for a in v['action_space']]
                         for v in config['site'].values()]
            actions = [reduce(lambda x,y:x*y,action)
                         for action in actions]
            args['action_space'] = reduce(lambda x,y:x*y,actions)
        args['action_table'] = self.get_action_table(if_mac)
        return args
