# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:52:04 2022

@author: MOMO
"""
from .environment_base import env_base
import pyswmm.toolkitapi as tkai
from struct import pack

class env_ast(env_base):
    def __init__(self, config, ctrl=True, binary=None):
        super().__init__(config, ctrl, binary)

        self.methods.update({'getnumobjects':self._getNumObjects,
                             'getflowunit':self._getFlowUnit,
                             'getobjectid':self._getObjectId,
                             'runoffS':self._getSubcatchRunoff,
                             'infilS':self._getSubcatchInfil,
                             'isstorage':self._is_Storage,
                             'lateralinflowN':self._getNodeLateralinflow,
                             'setting':self._getLinkSetting,
                             'cumflooding':self._getCumFlooding,
                             'totalinflow':self._getNodeTotalInflow,
                             'rainfall':self._getGageRainfall,
                             'getlinktype':self._getLinkType})

    def save_hotstart(self,hsf_file):
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


    # For hotstart file
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

    # For performance Target
    def _getCumFlooding(self,ID):
        if ID == "system":
            return self.sim._model.flow_routing_stats()['flooding']
        else:
            return self.sim._model.node_statistics(ID)['flooding_volume']
    
    def _getNodeTotalInflow(self,ID):
        # Cumulative inflow volume
        return self.sim._model.node_inflow(ID)


    def _getGageRainfall(self,ID):
        # For Cumrainfall state
        return self.sim._model.getGagePrecip(ID)[
            tkai.RainGageResults.rainfall.value]

    def _getLinkType(self,ID):
        # For control formulation
        return self.sim._model.getLinkType(ID).name
