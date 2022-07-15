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
#    if exists(env.filedir)==False:
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
    ax.set_ylabel('Rainfall Volume (mm)',fontsize=16)
    ax2 = ax.twinx()
    ax.invert_yaxis()
    for col in columns:
        ax2.plot(results[col].index,results[col],label=col,zorder=2)
    ax2.set_title(datestr,fontsize=16)
    ax2.set_xlabel('Time (H:M)',fontsize=16)
    ax2.set_ylabel('CSO ($\mathregular{m^3/s}$)',fontsize=16)
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')    
    if len(results)>=288:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(labelsize=14)
    ax2.legend(loc = 'upper right',fontsize=14)
    ax.legend(loc='upper left',fontsize=14)
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








       
    
    
    
    
    