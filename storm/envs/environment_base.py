# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from pystorms.environment import environment
import numpy as np

class env_base(environment):
    """Environment subclassed from the original form in pystorms
    
    This class acts as an interface between swmm's simulation
    engine and computational components. This class's methods are defined
    as getters and setters for generic stormwater attributes. So that, if need be, this
    class can be updated with a different simulation engine, keeping rest of the
    workflow stable.
    
    Attributes
    ----------
    config : dict
        dictionary with swmm_ipunt and, action and state space `(ID, attribute)`
    ctrl : boolean
        if true, config has to be a dict, else config needs to be the path to the input file
    binary: str
        path to swmm binary; this enables users determine which version of swmm to use

    Methods
    ----------
    step
        steps the simulation forward by a time step and returns the new state
    initial_state
        returns the initial state in the stormwater network
    terminate
        closes the swmm simulation
    reset
        closes the swmm simulaton and start a new one with the predefined config file.
    """
    def __init__(self, config, ctrl=True, binary=None):
        super().__init__(config, ctrl, binary)
        self._isFinished = False
        self._advance_seconds = None
    
    def step(self, actions=None, advance_seconds = None):
        r"""
        Implements the control action and forwards
        the simulation by a step.

        Parameters:
        ----------
        actions : list or array of dict
            actions to take as an array (1 x n)
        advance_seconds : int
            seconds for swmm to stride forward
        Returns:
        -------
        done : boolean
            event termination indicator
        """

        if (self.ctrl) and (actions is not None):
            # implement the actions based on type of argument passed
            # if actions are an array or a list
            if type(actions) == list or type(actions) == np.ndarray:
                for asset, valve_position in zip(self.config["action_space"], actions):
                    self._setValvePosition(asset, valve_position)
            elif type(actions) == dict:
                for valve_position, asset in enumerate(actions):
                    self._setValvePosition(asset, valve_position)
            else:
                raise ValueError(
                    "actions must be dict or list or np.ndarray \n got{}".format(
                        type(actions)
                    )
                )
        
        self._advance_seconds = advance_seconds
        # take the step !
        # add the swmm_stride option for a longer control step
        if self._advance_seconds is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self._advance_seconds)
        done = False if time > 0 else True
        return done

    def terminate(self):
        r"""
        Terminates the simulation
        """
        super().terminate()
        self._isFinished = True

    def reset(self):
        r"""
        Resets the simulation and returns the initial state

        Returns
        -------
        initial_state : array
            initial state in the network

        """
        state = super().reset()
        self._isFinished = False
        return state
