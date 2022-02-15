# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:27:38 2021

@author: chong
"""
from env_swmm_real import Ast
from Cen_RL import Cen_RL
from vdn import VDN
from iql import IQL
# import tqdm
from os import listdir
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.dates as mdates

import random
import pandas as pd
from datetime import datetime,timedelta
from swmm_api import read_out_file,read_rpt_file,read_inp_file
plt.rc('font',family = 'Times New Roman')


# init SWMM environment and agents
env = Ast(inp_test_file='./test/Real_{0}_{1}.inp')
cen_rl = Cen_RL(action_size = env.action_size)
cen_rl.load(model_dir='./model/DQN/')
vdnn = VDN(action_size = env.action_size,RGs = [2,0,2,1])
vdnn.load(model_dir='./model/VDN/')
iqll = IQL(action_size = env.action_size,RGs = [2,0,2,1])
iqll.load(model_dir='./model/IQL/')


def read_flooding(files,labels,cumulative = False):
    outfiles = [inpfile.replace('.inp','.out') for inpfile in files]
    flooding = pd.DataFrame()
    floods = {}
    res = {}
    for idx,out_file in enumerate(outfiles):
        out = read_out_file(out_file)
        flood = out.get_part('system')['Flow_lost_to_flooding']
        out.close()
        if cumulative:
            flood = (flood*5*60/1000).cumsum()
        floods[labels[idx]] = round(flood.sum()*5*60/1000,3)
        flooding[labels[idx]] = flood
        
        rpt = read_rpt_file(out_file.replace('.out','.rpt'))
        node_flood = rpt.node_flooding_summary
        if len(node_flood) == 0:
            res[labels[idx]] = 1
        else:
            inflow = rpt.flow_routing_continuity['Dry Weather Inflow']['Volume_10^6 ltr'] + rpt.flow_routing_continuity['Wet Weather Inflow']['Volume_10^6 ltr']
            res[labels[idx]] = 1-sum(node_flood['Total_Flood_Volume_10^6 ltr']/inflow*node_flood['Hours_Flooded']/24)
    return floods,res,flooding
    
events = pd.read_csv(env.event_file)
events = events[events['date'].apply(lambda date:int(date[-1])>=7)]
events = events[events['Precip']>20]
events = events[events['Precip']<60]
times = [(s,e) for s,e in zip(events['Start'],events['End'])]
test_datestrs = [ti[0][:10] for ti in times]


columns = ['BC','IQL','DQN','VDN']
test_floodings = pd.DataFrame(columns = ['Date','Start_Time','End_Time','Duration','Precipitation']+columns)


plt.tight_layout(pad=0.4,w_pad=0.5)
mpl.use('Agg')
for idx,time in enumerate(times):
    datestr = test_datestrs[idx]
    
    # Simulation
    dt_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
    for f in [iqll,cen_rl,vdnn]:
        filedir = env.inp_test_file.format(f.name,str(dt_time.date())+'-'+str(dt_time.hour))
        if exists(filedir.replace('inp','out')) and exists(filedir.replace('inp','rpt')):
            continue
        _,_,_ = env.test(f,time,filedir=filedir) 
    filedir = env.inp_test_file.format('BC',str(dt_time.date())+'-'+str(dt_time.hour))
    if exists(filedir.replace('inp','out')) == False or exists(filedir.replace('inp','rpt')) == False:
        bc_reward = env.test_bc_sing(time)
    print('Finish Simulation: {0}-{1}'.format(datestr,str(dt_time.hour)))

    # Get rain data
    inp = read_inp_file(filedir)
    rains = pd.DataFrame()
    for k,v in inp.TIMESERIES.items():
        rain = v.frame
        rains[k] = rain
    # Get flood data
    files = [env.inp_test_file.format(name,str(dt_time.date())+'-'+str(dt_time.hour)) 
                          for name in ['BC',iqll.name,cen_rl.name,vdnn.name]]
    floods,_,floodings = read_flooding(files,labels=columns,cumulative=False)
    results = pd.merge(rains,floodings,left_index=True,right_index=True)
    
    # Plot
    fig = plt.figure(figsize=(15,5),dpi=600)
    ax = fig.add_subplot(1,1,1)
    for col in list(inp.TIMESERIES.keys()):
        ax.bar(results[col].index,results[col],label=col,width=0.003,alpha=0.6,zorder=1)
    ax.set_ylabel('Rainfall Volume (mm)')
    ax2 = ax.twinx()
    ax.invert_yaxis()
    for col in columns:
        ax2.plot(results[col].index,results[col],label=col,zorder=2)
    ax2.set_title(datestr)
    ax2.set_xlabel('Time (H:M)')
    ax2.set_ylabel('CSO ($\mathregular{m^3/s}$)')
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')    
    if len(results)>=288:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.legend(loc = 'upper right')
    ax.legend(loc='upper left')
    plt.savefig('./test/results/'+datestr.replace('/','_')+'_'+str(dt_time.hour)+'.png')
    plt.close('all')
    print('Finish Plot: {0}-{1}'.format(datestr,str(dt_time.hour)))
    
    floods['Date'] = dt_time.strftime('%m/%d/%Y')
    floods['Start_Time'] = time[0]
    floods['End_Time'] = time[1]
    floods['Duration'] = (datetime.strptime(time[1],'%m/%d/%Y %H:%M:%S')-datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')).total_seconds()/60
    rpt = read_rpt_file(filedir.replace('.inp','.rpt'))
    floods['Precipitation'] = rpt.runoff_quantity_continuity['Total Precipitation']['Depth_mm']
    test_floodings.loc[idx,floods] = floods
test_floodings.round(3).to_csv('./test/results/floodings.csv')






# Process: tank depths & orifice settings
# event in 10/01/2009

columns = ['DQN','IQL','VDN']
time = ('10/01/2009 14:00:00','')
dt_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
files = []
for f in [cen_rl,iqll,vdnn]:
    env.filedir = env.inp_test_file.format(f.name,str(dt_time.date())+'-'+str(dt_time.hour))
    # if exists(env.filedir.replace('inp','out'))==False:
    test_reward,acts,_ = env.test(f,time,filedir=env.filedir)
    files.append(env.filedir)
inp = read_inp_file(files[-1])
rains = pd.DataFrame()
for k,v in inp.TIMESERIES.items():
    rain = v.frame
    rains[k] = rain
RG = {2:[('rain3','#2ca02c')],3:[('rain1','#1f77b4'),('rain2','#d62728')],
      4:[('rain3','#2ca02c')],6:[('rain2','#d62728')]}
fig = plt.figure(figsize=(30,10),dpi = 600)
objs = []
for idx,i in enumerate([2,3,4,6]):
    depths = []
    settings = []
    for file in files:
        out = read_out_file(file.replace('inp','out'))
        depths.append(out.get_part('node','T%s'%i)['Depth_above_invert'])
        settings.append(out.get_part('link','V%s'%i)['Capacity'])
        out.close()
    
    ax = fig.add_subplot(2,4,idx+1)
    ax.set_title('T%s'%i,fontsize=16)
    for col in RG[i]:
        ba = ax.bar(rains.index,rains[col[0]],
                    label=col[0],color=col[1],width=0.003,alpha=0.5,zorder=1)
        if idx<=1:
            objs.append(ba)
    ax2 = ax.twinx()
    ax.invert_yaxis()
    ax.set_ylim((rains.max().max()*2,0))
    
    

    time = settings[0].index.tolist()
    time = [t+timedelta(minutes=i) for t in time for i in range(5)]
    capa = [[ca for ca in setting.values.tolist() for i in range(5)] 
            for setting in settings]
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    
    ax2.plot(time,capa[0],'-',label = 'DQN')
    ax2.plot(time,capa[1],'--',label = 'IQL')  
    ax2.plot(time,capa[2],'-.',label = 'VDN')  
    ax2.set_ylim([0,1.2])    
    
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    if idx==0:
        ax.set_ylabel('Rainfall Volume (mm)',fontsize=16)
    if idx==3:
        ax2.set_ylabel('Orifice Setting',fontsize=16)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    
    
    ax = fig.add_subplot(2,4,idx+5)
    for col in RG[i]:
        ba = ax.bar(rains.index,rains[col[0]],
                    label=col[0],color=col[1],width=0.003,alpha=0.5,zorder=1)
    ax2 = ax.twinx()
    ax.invert_yaxis()
    ax.set_ylim((rains.max().max()*2,0))

    time = depths[0].index.tolist()
    dep = [depth.values.tolist()
           for depth in depths]
    
    depth = ax2.plot(time,dep[0],'-',label = 'DQN')
    depth2 = ax2.plot(time,dep[1],'--',label = 'IQL')
    depth3 = ax2.plot(time,dep[2],'-.',label = 'VDN')
    ax2.set_ylim([0,5.1])    
    
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    if idx==0:
        ax.set_ylabel('Rainfall Volume (mm)',fontsize=16)
    if idx==3:
        ax2.set_ylabel('Tank Depth (m)',fontsize=16)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
objs.sort(key=lambda x:x.get_label())
objs += depth+depth2+depth3
fig.legend(objs,[l.get_label() for l in objs],loc=8,ncol=6,fontsize=16)
fig.savefig('./test/results/process.png')    

    
    
    