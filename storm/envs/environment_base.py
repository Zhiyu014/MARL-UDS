# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from pystorms.environment import environment
import numpy as np
from struct import pack
import pyswmm.toolkitapi as tkai
from pyswmm.swmm5 import PySWMM

class env_base(environment):
    """Environment subclassed from the original environment in pystorms

    Added: 
    ----------
    swmm_stride
    save_hotstart
    _isFinished

    
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
    save_hotstart
        generate a hsf file of the current hydraulic and hydrodynamic status.
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
        if not self._isFinished:
            self.terminate()

        # Start the next simulation
        self.sim._model = PySWMM(self.config["swmm_input"])
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()

        # get the state
        state = self._state()
        self._isFinished = False
        return state


    def save_hotstart(self,hsf_file):
        r"""
        Outputs a hotstart file and returns file path.

        Parameters:
        ----------
        hsf_file : str, path-like
            the hsf file path to generate
        Returns
        -------
        initial_state : array
            initial state in the network

        """
        filestamp = 'SWMM5-HOTSTART4'
        with open(hsf_file,'wb') as f:
            f.write(bytes(filestamp,encoding='utf-8'))
            for col in ['SUBCATCH','LANDUSE','NODE','LINK','POLLUT']:
                f.write(pack('i',self._getNumObjects(col)))
            f.write(pack('i',['CFS','GPM','MGD','CMS','LPS','MLD'].index(self._getFlowUnit())))    #FlowUnits
            
            for idx in range(self._getNumObjects('SUBCATCH')):
                _subcatchmentid = self._getObjectId('SUBCATCH',idx)
                runoff = self._getSubcatchRunoff(_subcatchmentid)
                infiltration_loss = self._getSubcatchInfil(_subcatchmentid)
                x = (0.0,0.0,0.0,runoff)  # ponded depths in 3 subareas, runoff
                f.write(pack('dddd',*x))
                x = (0.0,infiltration_loss,0.0,0.0,0.0,0.0)
                f.write(pack('dddddd',*x))
                
            for idx in range(self._getNumObjects('NODE')):
                _nodeid = self._getObjectId('NODE',idx)
                depth = self.methods['depthN'](_nodeid)
                lateral_inflow = self._getNodeLateralinflow(_nodeid)
                x = (depth,lateral_inflow)
                f.write(pack('ff',*x))
                if self._is_Storage(_nodeid):
                    f.write(pack('f',0)) # no api for the HRT of storage
            
            for idx in range(self._getNumObjects('LINK')):
                _linkid = self._getObjectId('LINK',idx)
                x = (self.methods['flow'](_linkid),self.methods['depthL'](_linkid))
                x += (self._getLinkSetting(_linkid),)
                f.write(pack('fff',*x))
        return hsf_file


    # ------ For hotstart file  ----------------------------------------------
    def _getNumObjects(self,__type):
        __type = getattr(tkai.ObjectType,__type.upper(),'value')
        return self.sim._model.getProjectSize(__type)

    def _getFlowUnit(self):
        return self.sim._model.getSimUnit(tkai.SimulationUnits.FlowUnits.value)

    def _getObjectId(self,__type,idx):
        __type = getattr(tkai.ObjectType,__type.upper(),'value')
        return self.sim._model.getObjectId(__type,idx)

    def _getSubcatchRunoff(self,_subcatchmentid):
        return self.sim._model.getSubcatchResult(_subcatchmentid,
                                                 tkai.SubcResults.newRunoff.value)

    def _getSubcatchInfil(self,_subcatchmentid):
        return self.sim._model.getSubcatchResult(_subcatchmentid,
                                                 tkai.SubcResults.infilLoss.value)

    def _is_Storage(self,_nodeid):
        return self.sim._model.getNodeType(_nodeid) is tkai.NodeType.storage.value

    def _getNodeLateralinflow(self,_nodeid):
        return self.sim._model.getNodeResult(_nodeid,
                                            tkai.NodeResults.newLatFlow.value)

    def _getNodeLateralinflow(self,_nodeid):
        return self.sim._model.getNodeResult(_nodeid,
                                            tkai.NodeResults.newLatFlow.value)

    def _getLinkSetting(self,_linkid):
        return self.sim._model.getLinkResult(_linkid,
                                            tkai.LinkResults.setting.value)