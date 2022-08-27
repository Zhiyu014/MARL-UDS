from pystorms.scenarios import scenario
from .environment_astlingen import env_ast
import os
import yaml
import numpy as np
from swmm_api import read_inp_file
from swmm_api.input_file.sections import FilesSection,Control
import datetime
from functools import reduce
from itertools import product

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

    def __init__(self, config_file=None, swmm_file=None, if_predict=False):
        # Network configuration
        config_file = os.path.join(HERE,"config","astlingen.yaml") \
            if config_file is None else config_file
        self.config = yaml.load(open(config_file, "r"), yaml.FullLoader)
        self.config["swmm_input"] = os.path.join(HERE,"network",self.config["env_name"],self.config["env_name"] +'.inp') \
            if swmm_file is None else swmm_file
        
        # Create the environment based on the physical parameters
        self.env = env_ast(self.config, ctrl=True)
        
        self.penalty_weight = {ID: weight
                               for ID, _, weight in \
                                   self.config["performance_targets"]}
        self.if_predict = if_predict # Not sure yet

        # initialize logger
        self.initialize_logger()


    def step(self, actions=None, advance_seconds = None, log=True):
        # Implement the actions and take a step forward
        if advance_seconds is None and 'control_interval' in self.config:
            advance_seconds = self.config['control_interval'] * 60
        if actions is not None:
            actions = self._convert_actions(actions)

        done = self.env.step(actions, advance_seconds = advance_seconds)
        
        # Log the states, targets, and actions
        if log:
            self._logger()

        # Log the performance
        self._calc_perf_value()

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
            if attribute in ['depthN','rainfall']:
                state.append(__state[idx])
            else:
                if len(self.data_log[attribute][ID]) > 1:
                    __value = __state[idx] - self.data_log[attribute][ID][-2]
                    state.append(__value)
                else:
                    state.append(__state[idx])
        state = np.asarray(state)
        return state

    def reset(self,swmm_file=None):
        # clear the data log and reset the environment
        if swmm_file is None:
            _ = self.env.reset()
            state = self.state()
        else:
            # change the swmm inp file
            self.config["swmm_input"] = swmm_file
            self.env = env_ast(self.config, ctrl=True)
            state = self.state()
        self.initialize_logger()
        return state
        
    def initialize_logger(self, config=None):
        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "setting": {}
        }
        config = self.config if config is None else config
        # Data logger for storing _performance & _state data
        for ID, attribute, _ in config["performance_targets"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []
            
        for ID, attribute in config["states"]:
            if attribute not in self.data_log.keys():
                self.data_log[attribute] = {}
            self.data_log[attribute][ID] = []

        for ID in config["action_space"].keys():
            self.data_log["setting"][ID] = []
        
        # if self.if_predict:
        #     self.data_log.update({"hotstart_file": [],
        #     "evaluation_file": []})

    def _convert_actions(self,actions):
        if actions is not None:
            if type(actions) == list or type(actions) == np.ndarray:
                if {type(a) for a in actions} == {int}:
                    actions = [options[actions[idx]]
                    for idx,options in enumerate(self.config['action_space'].values())]
            elif type(actions) == dict and {type(v) for v in actions.values()} == {int}:
                actions = [self.config['action_space'][k][v]
                for k,v in actions.items()]
        return actions

    def _logger(self):
        super()._logger()
        # for attribute in self.data_log.keys():
        #     if attribute not in ["performance_measure", "simulation_time",
        #     "hotstart_file", "evaluation_file"]:
        #         for element in self.data_log[attribute].keys():
        #             self.data_log[attribute][element].append(
        #                 self.env.methods[attribute](element)
        #             )
        # self.data_log["simulation_time"].append(self.env.methods['simulation_time']())

        # if self.if_predict:
        #     hsf_file = self.save_hotstart()
        #     eval_file = self.create_eval_file(hsf_file)
        #     self.data_log["hotstart_file"].append(hsf_file)
        #     self.data_log["evaluation_file"].append(eval_file)

        
    def _calc_perf_value(self):
        # Calculate the target error in the recent step based on cumulative values
        __performance = 0.0
        for ID, attribute, weight in self.config["performance_targets"]:
            __cumvolume = self.env.methods[attribute](ID)
            # Recent volume has been logged
            if len(self.data_log[attribute][ID]) > 1:
                __volume = __cumvolume - self.data_log[attribute][ID][-2]
            else:
                __volume = __cumvolume

            if attribute == "totalinflow" and ID != "Out_to_WWTP":
                if len(self.data_log[attribute][ID]) > 1:
                    __prevolume = np.diff([0]+self.data_log[attribute][ID])[-2]
                else:
                    __prevolume = 0
                __volume = abs(__volume - __prevolume)
            # __weight = self.penalty_weight[ID]
            __performance += __volume * weight

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

    def get_action_table(self,if_mac):
        action_table = {}
        actions = [len(v) for v in self.config['action_space'].values()]
        site_combs = product(*[range(act) for act in actions])
        for idx,site in enumerate(site_combs):
            if if_mac:
                action_table[site] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
            else:
                action_table[idx] = [v[site[i]]
                    for i,v in enumerate(self.config['action_space'].values())]
        return action_table

    def get_args(self,if_mac = True):
        args = self.config.copy()
        # Rainfall timeseries & events files
        args['rainfall_timeseries'] = os.path.join(HERE,'config',args['rainfall_timeseries']+'.csv')
        args['rainfall_events'] = os.path.join(HERE,'config',args['rainfall_events']+'.csv')
        # Control interval for step_advance
        args['state_shape'] = len(args['states'])
        if if_mac:
            # multi-agent controller structure
            args["n_agents"] = len(args['site'])

            # Specify the observe data for each site
            state = [s[0] for s in args['states']]
            args['observ_space'] = [[state.index(o) for o in v['states']]
            for v in args['site'].values()]

            args['action_space'] = [len(v['action_space']) for v in args['site'].values()]

        else:
            args['n_agents'] = 1

            args['observ_space'] = args['state_shape']  # int value
            
            actions = [len(v) for v in args['action_space'].values()]
            args['action_space'] = reduce(lambda x,y:x*y,actions)
        args['action_table'] = self.get_action_table(if_mac)
        return args

    # predictive functions
    def save_hotstart(self,hsf_file=None):
        # Save the current state in a .hsf file.
        if hsf_file is None:
            ct = self.env.methods['simulation_time']()
            hsf_file = '%s.hsf'%ct.strftime('%Y-%m-%d-%H-%M')
            hsf_file = os.path.join(os.path.dirname(self.config['swmm_input']),
            self.config['hsf_dir'],hsf_file)
        if os.path.exists(os.path.dirname(hsf_file)) == False:
            os.mkdir(os.path.dirname(hsf_file))
        return self.env.save_hotstart(hsf_file)

    def create_eval_file(self,hsf_file=None):
        ct = self.env.methods['simulation_time']()
        inp = read_inp_file(self.config['swmm_input'])

        # Set the simulation time & hsf options
        inp['OPTIONS']['START_DATE'] = inp['OPTIONS']['REPORT_START_DATE'] = ct.date()
        inp['OPTIONS']['START_TIME'] = inp['OPTIONS']['REPORT_START_TIME'] = ct.time()
        inp['OPTIONS']['END_DATE'] = (ct + datetime.timedelta(minutes=self.config['eval_horizon'])).date()
        inp['OPTIONS']['END_TIME'] = (ct + datetime.timedelta(minutes=self.config['eval_horizon'])).time()
        
        if hsf_file is not None:
            if 'FILES' not in inp:
                inp['FILES'] = FilesSection()
            inp['FILES']['USE HOTSTART'] = hsf_file
        
        # Set the Control Rules
        inp['CONTROLS'] = Control.create_section()
        for i in range(self.config['control_horizon']//self.config['control_interval']):
            time = round(self.config['control_interval']/60*(i+1),2)
            conditions = [Control._Condition('IF','SIMULATION','TIME', '<', str(time))]
            actions = []
            for idx,k in enumerate(self.config['action_space']):
                logic = 'THEN' if idx == 0 else 'AND'
                kind = self.env.methods['getlinktype'](k)
                action = Control._Action(logic,kind,k,'SETTING','=',str('1.0'))
                actions.append(action)
            inp['CONTROLS'].add_obj(Control('P%s'%(i+1),conditions,actions,priority=5-i))
    

        # Output the eval file
        eval_inp_file = os.path.join(os.path.dirname(self.config['swmm_input']),
                                self.config['eval_dir'],
                                self.config['suffix']+os.path.basename(self.config['swmm_input']))
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

