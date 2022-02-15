# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:54:10 2021

@author: MOMO
"""

from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api import read_inp_file,swmm5_run,read_rpt_file,read_out_file
from swmm_api.input_file.sections.others import TimeseriesData
import random
from datetime import datetime,timedelta
from math import log10
from numpy import std,average,argmax,array,logspace,multiply,identity
import pandas as pd
from os.path import exists
# from networkx import Graph,adjacency_matrix

# REWARD_DISCOUNT = 0.1

class Ast:
    def __init__(self,inp_file = './Astlingen.inp',
                 inp_train_file = './train/Real_{0}_{1}.inp',
                 inp_test_file = './test/Real_{0}_{1}.inp',
                 donothing_inp = './donothing.inp',
                 BC_inp = './Astlingen_SWMM.inp',
                 EFD_inp = './Real_EFDC.inp',
                 rain_file = './rainfiles.csv',
                 date_file = './date_sum.csv',
                 event_file = './rains_2h.csv',
                 flood_nodes = None):
        self.n_agents = 4
        self.n_raingages = 4
        self.control_step = 300
        # self.duration = 2*3600
        # self.timesteps = self.duration/self.control_step+1
        self.action_size = 3
        self.observ_size = 16
        self.update_num = 12*24
        self.orfices = ['V2','V3','V4','V6']
        self.tanks = ['T2','T3','T4','T6']
        self.in_nodes = ['J13','J12','J5','J8']
        self.out_nodes = ['J17','J14','J6','J9']
        self.flood_nodes = flood_nodes
        self.actions = {'V2':[0.1075,0.2366,1.0],'V3':[0.3159,0.6508,1.0],
                        'V4':[0.1894,0.3523,1.0],'V6':[0.1687,0.4303,1.0]}
        
        self.do_nothing = donothing_inp
        self.BC_inp = BC_inp
        self.EFD_inp = EFD_inp
        self.inp_file = inp_file
        self.inp_train_file = inp_train_file
        self.inp_test_file = inp_test_file
        # inp = read_inp_file(inp_file)
        
        # self.raindatas = ['./{0}Astlingen_Erft{1}.txt'.format(i+1,i+1) for i in range(self.n_raingages)]
        # files = pd.concat([pd.read_csv(self.raindatas[i],header=None,names=['date','time','RG%s'%(i+1)],sep='\s+') for i in range(self.n_raingages)],axis=1)
        # files = files.loc[:,~files.columns.duplicated()]
        # files.to_csv('./rainfiles.csv')
        
        # date_sum = files.groupby("date").agg('sum').sum(axis=1)
        # self.date_sum = date_sum[date_sum!=0]
        # self.date_sum = pd.read_csv(date_file)
        self.rain_file = rain_file
        self.event_file = event_file

    def random_sample(self,rain_num,a,b,yr,train=True):
        if train:
            date_sum = self.date_sum[self.date_sum['date'].apply(lambda date:int(date[-1])<yr)] #New added: Filter
        else:
            date_sum = self.date_sum[self.date_sum['date'].apply(lambda date:int(date[-1])>=yr)] #New added: Filter
        date_sum = date_sum.set_index('date')
        date_sum = date_sum[date_sum['0']>a]
        date_sum = date_sum[date_sum['0']<b]
        date_sum['weights'] = date_sum['0'].apply(lambda i:i/date_sum['0'].sum())
        dates = list(date_sum.sample(rain_num,weights=date_sum['weights'])['0'].to_dict())
        dates = [(int(date.split('/')[-1]),int(date.split('/')[0]),int(date.split('/')[1])) for date in dates]
        return dates

    def rand_events(self,rain_num,a,b,yr,train=True):
        events = pd.read_csv(self.event_file)
        if train:
            events = events[events['date'].apply(lambda date:int(date[-1])<yr)] #New added: Filter
        else:
            events = events[events['date'].apply(lambda date:int(date[-1])>=yr)] #New added: Filter
        events = events[events['Precip']>a]
        events = events[events['Precip']<b]
        
        # date_sum['weights'] = date_sum['0'].apply(lambda i:i/date_sum['0'].sum())
        samples = events.sample(rain_num)
        times = [(s,e) for s,e in zip(samples['Start'],samples['End'])]
        return times
    
    # @ todo split files
    def generate_file(self,times,name='VDN',train = True):
        inp = read_inp_file(self.inp_file)
        files = pd.read_csv(self.rain_file,index_col=0)
        files['datetime'] = files['date']+' '+files['time']
        files['datetime'] = files['datetime'].apply(lambda dt:datetime.strptime(dt, '%m/%d/%Y %H:%M:%S'))
        
        for idx,(start,end) in enumerate(times):
            filedir = self.inp_train_file.format(name,idx) if train else self.filedir
            
            if exists(filedir) == True:
                continue
            start_time = datetime.strptime(start,'%m/%d/%Y %H:%M:%S')
            end_time = datetime.strptime(end,'%m/%d/%Y %H:%M:%S')+timedelta(hours=2)            
            rain = files[start_time<files['datetime']]
            rain = rain[rain['datetime']<end_time]
            raindata = [[[date+' '+time,intense] for date,time,intense in zip(rain['date'],rain['time'],rain['RG%s'%(i+1)])] for i in range(self.n_raingages)]
            for idx,k in enumerate(inp.TIMESERIES.keys()):
                inp.TIMESERIES[k] = TimeseriesData('rain%s'%(idx+1),raindata[idx])
            inp.OPTIONS['START_DATE'] = start_time.date()
            inp.OPTIONS['END_DATE'] = end_time.date()
            inp.OPTIONS['START_TIME'] = start_time.time()
            inp.OPTIONS['END_TIME'] = end_time.time()
            inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
            inp.OPTIONS['REPORT_START_TIME'] = start_time.time()
            inp.write_file(filedir)
        
    
    
    def run_simulation(self,f,start = None, end = None, train=True,
                       sensi = None,fail=None,act_fail=None,
                       inte_fail=None,f1=None,f2=None):
        states = []
        actions = []
        rewards = []
        with Simulation(self.filedir) as sim:
            if start != None:
                sim.start_time = start
            if end != None:
                sim.end_time = end
            nodes = Nodes(sim)
            links = Links(sim)
            sys = SystemStats(sim)
            rgs = RainGages(sim)
            cum_inflow = 0
            cum_outfall = 0
            if self.flood_nodes is not None:
                cum_floods = [0 for _ in range(len(self.flood_nodes))] 
            else:
                cum_floods = 0
            for st in sim:
                sim.step_advance(self.control_step)
                precip = [rgs['RG1'].rainfall,rgs['RG2'].rainfall,rgs['RG3'].rainfall,rgs['RG4'].rainfall]
                    
                #cum_outfall.append(nodes['Out_to_WWTP'].total_inflow)
                #cum_out_sigma = std(cum_outfall)
                #if cum_out_sigma == 0:
                #    cum_out_sigma =1
                
                depth = [nodes[tank].depth/nodes[tank].full_depth for tank in self.tanks]
                in_depth = [nodes[node].depth/nodes[node].full_depth for node in self.in_nodes]
                out_depth = [nodes[node].depth/nodes[node].full_depth for node in self.out_nodes]
                
                routing = sys.routing_stats
                
                inflow = routing['wet_weather_inflow']+routing['dry_weather_inflow']-cum_inflow
                cum_inflow = routing['wet_weather_inflow']+routing['dry_weather_inflow']
                
                if self.flood_nodes is not None:
                    flood = [sum([nodes[n].statistics['flooding_volume']
                              for n in node])-cum_floods[j] 
                             for j,node in enumerate(self.flood_nodes)]           
                    cum_floods = [sum([nodes[n].statistics['flooding_volume'] 
                                       for n in node])
                                  for node in self.flood_nodes]
                    reward = [-fl/inflow for fl in flood]
                else:
                    flood = routing['flooding'] - cum_floods
                    cum_floods = routing['flooding']
                    reward = -flood/inflow
                    
                outfall = routing['outflow'] - cum_outfall
                cum_outfall = routing['outflow']


                          
                if fail is not None:
                    is_fail = [int(random.random()>fail) for _ in range(self.n_agents)]
                    depth = [j*is_fail[i] for i,j in enumerate(depth)]       
                    in_depth = [j*is_fail[i] for i,j in enumerate(in_depth)]       
                    out_depth = [j*is_fail[i] for i,j in enumerate(out_depth)]       

                state = precip + in_depth + out_depth + depth
                
                if sensi is not None:
                    state = [max(random.gauss(j,sensi*j),0) for j in state]          
                
                action = f.act(state,train)

                if inte_fail is not None:
                    is_fail = [int(random.random()>inte_fail) for _ in range(self.n_agents)]
                    action1 = f1.act(state,train=False)
                    action2 = f2.act(state,train=False)
                    action = array(action1)*array(is_fail) + array(action2)*(1-array(is_fail))
                    action = action.tolist()

                    
                if act_fail is not None:
                    is_fail = [int(random.random()>act_fail) for _ in range(self.n_agents)]
                
                for i,orf in enumerate(self.orfices):                    
                    if act_fail is not None and is_fail[i]==0:
                        continue
                    links[orf].target_setting = self.actions[orf][action[i]]
     
                states.append(state)      # observs (observ_size,n_agents,timesteps)
                rewards.append(reward)    # rewards  (1,timesteps)
                actions.append(action)      # actions (action_size,n_agents,timesteps)
        next_states = states[1:]
        states,rewards,actions = states[:-1],rewards[1:],actions[:-1]
        self.update_num = len(actions)
        return states,next_states,rewards,actions,cum_inflow

        
    def test(self,f,time,sensi=None,fail=None,act_fail=None,inte_fail=None,f1=None,f2=None,filedir=None):

        start_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
        
        if filedir == None:            
            self.filedir = self.inp_test_file.format(f.name,str(start_time.date())+'-'+str(start_time.hour))
        else:
            self.filedir = filedir
        if exists(self.filedir) == False:
            self.generate_file([time],name=f.name,train=False)
        observs,next_observs,rewards,actions,floods = self.run_simulation(f,train=False,sensi=sensi,fail=fail,act_fail=act_fail,
                                                                              inte_fail=inte_fail,f1=f1,f2=f2)
        loss = f._test_loss(observs,next_observs,rewards,actions)
        score = self.read_rpt(self.filedir.replace('inp','rpt'))
        print('Control Score:   %s'%score)
        return score,actions,loss
    
    def test_dura(self,f,start,end):
        self.filedir = self.inp_file
        observs,next_observs,states,next_states,rewards,actions,_ = self.run_simulation(f.agents,start = start,end = end, train=False)
        score = self.read_rpt(self.inp_file.replace('inp','rpt'))
        print('Control Score:   %s'%score)
        return score
            
    def read_rpt(self,rpt_path):
        rpt = read_rpt_file(rpt_path)
        flooding = rpt.flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
        cum_inflow = rpt.flow_routing_continuity['Wet Weather Inflow']['Volume_10^6 ltr']+rpt.flow_routing_continuity['Dry Weather Inflow']['Volume_10^6 ltr']
        if cum_inflow == 0:
            return 1
        return 1-flooding/cum_inflow
    
    def test_no(self,time):
        start_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
        end_time = datetime.strptime(time[-1],'%m/%d/%Y %H:%M:%S')+timedelta(hours=2)
        with Simulation(self.do_nothing) as sim:
            sim.start_time = start_time
            sim.end_time = end_time
            for st in sim:
                pass
        no_reward = self.read_rpt(self.do_nothing.replace('inp','rpt'))
        print('Do-nothing Reward:   %s'%no_reward)
        return no_reward
    
    def test_bc_sing(self,time):
        start_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
        self.filedir = self.inp_test_file.format('BC',str(start_time.date())+'-'+str(start_time.hour))
        if exists(self.filedir) == False:
            self.generate_file([time],name='BC',train=False)
        with Simulation(self.filedir) as sim:
            links = Links(sim)
            for st in sim:
                sim.step_advance(self.control_step)
                for li in self.orfices:
                    links[li].target_setting = self.actions[li][1]
                pass
        score = self.read_rpt(self.filedir.replace('inp','rpt'))
        print('Baseline Score:   %s'%score)
        return score
        
        
    def test_bc(self,time):
        start_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
        end_time = datetime.strptime(time[-1],'%m/%d/%Y %H:%M:%S')+timedelta(hours=2)      
        with Simulation(self.BC_inp) as sim:
            sim.start_time = start_time
            sim.end_time = end_time
            for st in sim:
                pass
        score = self.read_rpt(self.BC_inp.replace('inp','rpt'))
        print('Baseline Reward:   %s'%score)
        return score
    
    def test_efd(self,time):
        start_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
        end_time = datetime.strptime(time[-1],'%m/%d/%Y %H:%M:%S')+timedelta(hours=2)      
        with Simulation(self.EFD_inp) as sim:
            sim.start_time = start_time
            sim.end_time = end_time            
            # sim.start_time = datetime.datetime(*date,0,0)
            # sim.end_time = sim.start_time + datetime.timedelta(days=1)
            for st in sim:
                pass
        score = self.read_rpt(self.EFD_inp.replace('inp','rpt'))
        print('EFD Reward:   %s'%score)
        return score
    

  