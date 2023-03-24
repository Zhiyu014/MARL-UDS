# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from .environment_base import env_base
import pyswmm.toolkitapi as tkai

class env_chaohu(env_base):
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
        # Update some useful functions in chaohu scenario
        self.methods.update({'cumprecip':self._getSystemRainfall,
                             'cumflooding':self._getCumflooding,
                             'totalinflow':self._getNodeTotalInflow,
                             'lateral_infow_vol':self._getNodeLateralinflowVol,
                             'pumpenergy':self._getPumpEnergy,
                             'getnodefulldepth':self._getNodeFullDepth,
                             'getnodeinitdepth':self._getNodeInitDepth,
                             'setting':self._getLinkSetting,
                             'getlinktype':self._getLinkType})

    # ------ Get necessary Parameters  ----------------------------------------------
    def _getNodeTotalInflow(self,ID):
        # Cumulative inflow volume
        return self.sim._model.node_inflow(ID)
    
    def _getNodeLateralinflowVol(self,ID):
        # Cumulative lateral inflow volume
        return self.sim._model.node_statistics(ID)['lateral_infow_vol']
    
    def _getSystemRainfall(self,SysID):
        return self.sim._model.runoff_routing_stats()['rainfall']
    
    def _getCumflooding(self,ID):
        if ID == "system":
            return self.sim._model.flow_routing_stats()['flooding']
        else:
            return self.sim._model.node_statistics(ID)['flooding_volume']
    
    def _getNodeFullDepth(self,ID):
        return self.sim._model.getNodeParam(ID,tkai.NodeParams.fullDepth.value)

    def _getNodeInitDepth(self,ID):
        return self.sim._model.getNodeParam(ID,tkai.NodeParams.initDepth.value)

    def _getLinkSetting(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)    

    def _getPumpEnergy(self, ID):
        return self.sim._model.pump_statistics(ID)['energy_consumed']    

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name