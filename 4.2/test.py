# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:27:38 2021

@author: chong
"""
from env_swmm_real import Ast
from vdn import VDN
from qagent import QAgent
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd
from swmm_api import read_out_file,read_rpt_file,read_inp_file
from datetime import datetime
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999	


# init SWMM environment and agents
env = Ast()
agents = []
for i in range(env.n_agents):
    agent = QAgent(epsilon_decay,env.action_size,env.observ_size,dueling=True)
    agents.append(agent)
vdnn = VDN(agents=agents)
vdnn.load()

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

# times = env.rand_events(8,30,60,7,train=False)
# test_datestrs = [ti[0][:10] for ti in times]

# test_datestrs = ['07/18/2007','03/01/2008','02/28/2009','06/01/2009','11/14/2009']
# inps = [read_inp_file('./test/Real_VDN_'+date.split('/')[2]+'-'+date.split('/')[0]+'-'+date.split('/')[1]+'.inp')
#         for date in test_datestrs]
# times = [(inp.OPTIONS.START_DATE.strftime('%m/%d/%Y')+' '+inp.OPTIONS.START_TIME.strftime('%H:%M:%S'),
#           inp.OPTIONS.END_DATE.strftime('%m/%d/%Y')+' '+inp.OPTIONS.END_TIME.strftime('%H:%M:%S'))
#          for inp in inps]




columns = ['Uncontrolled','BC','EFD','VDN']
test_floodings = pd.DataFrame(columns = ['Start_Time','Precipitation']+columns)
test_res = pd.DataFrame(columns = ['Start_Time','Precipitation']+columns)


# fig = plt.figure(figsize=(15,25),dpi = 600)
plt.tight_layout(pad=0.4,w_pad=0.5)
mpl.use('Agg')
# fig2 = plt.figure(figsize = (10,5))
# ax2 = fig2.add_subplot(1,1,1)
for idx,time in enumerate(times):
    datestr = test_datestrs[idx]
    
    # Simulation
    dt_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
    env.filedir = env.inp_test_file.format(str(dt_time.date())+'-'+str(dt_time.hour))
    if exists(env.filedir)==False:
        test_reward,acts,_ = env.test(vdnn,time)
    no_reward = env.test_no(time)
    bc_reward = env.test_bc(time)
    efd_reward = env.test_efd(time)
    
    # Get rain data
    inp = read_inp_file(env.filedir)
    rains = pd.DataFrame()
    for k,v in inp.TIMESERIES.items():
        rain = v.frame
        rains[k] = rain
    # Get flood data
    files = [env.do_nothing,env.BC_inp,env.EFD_inp,env.filedir]
    floods,res,floodings = read_flooding(files,labels=columns,cumulative=False)
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
    plt.savefig('./test/figs/'+datestr.replace('/','_')+'_'+str(dt_time.hour)+'.png')
    plt.close('all')
    
    floods['Start_Time'] = dt_time.strftime('%m/%d/%Y %H:%M')
    res['Start_Time'] = dt_time.strftime('%m/%d/%Y %H:%M')
    rpt = read_rpt_file(env.filedir.replace('.inp','.rpt'))
    floods['Precipitation'] = rpt.runoff_quantity_continuity['Total Precipitation']['Depth_mm']
    res['Precipitation'] = rpt.runoff_quantity_continuity['Total Precipitation']['Depth_mm']
    test_floodings.loc[idx,floods] = floods
    test_res.loc[idx,res] = res
test_floodings.round(3).to_csv('./test/figs/floodings.csv')
test_res.round(3).to_csv('./test/figs/res.csv')

#
# import datetime
# start = datetime.datetime(2000,1,1,0,0)
# end = datetime.datetime(2009,12,31,23,55)
# vdn_score = env.test_dura(vdnn,start,end)
# no_score = env.test_no((start.strftime('%m/%d/%Y %H:%M:%S'),end.strftime('%m/%d/%Y %H:%M:%S')))
# bc_score = env.test_bc((start.strftime('%m/%d/%Y %H:%M:%S'),end.strftime('%m/%d/%Y %H:%M:%S')))
# efd_score = env.test_efd((start.strftime('%m/%d/%Y %H:%M:%S'),end.strftime('%m/%d/%Y %H:%M:%S')))



# Test all events
events = pd.read_csv(env.event_file)
#events = events[events['date'].apply(lambda date:int(date[-1])>=7)]

columns = ['Uncontrolled','BC','EFD','VDN']
test_floodings = pd.DataFrame(columns = columns)

for idx,time in enumerate(zip(events['Start'],events['End'])):

    datestr = time[0][:10]
    
    # Simulation
    dt_time = datetime.strptime(time[0],'%m/%d/%Y %H:%M:%S')
    env.filedir = env.inp_test_file.format(str(dt_time.date())+'-'+str(dt_time.hour))
    if exists(env.filedir)==False:
        test_reward,acts,_ = env.test(vdnn,time)
    no_reward = env.test_no(time)
    bc_reward = env.test_bc(time)
    efd_reward = env.test_efd(time)
    
    # Get rain data
    inp = read_inp_file(env.filedir)
    rains = pd.DataFrame()
    for k,v in inp.TIMESERIES.items():
        rain = v.frame
        rains[k] = rain
    # Get flood data
    files = [env.do_nothing,env.BC_inp,env.EFD_inp,env.filedir]
    floods,_,floodings = read_flooding(files,labels=columns,cumulative=False)
    results = pd.merge(rains,floodings,left_index=True,right_index=True)
    test_floodings.loc[idx,floods] = floods

    if exists('./test/figs/'+datestr.replace('/','_')+'_'+str(dt_time.hour)+'.png'):
        continue
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
    plt.savefig('./test/figs/'+datestr.replace('/','_')+'_'+str(dt_time.hour)+'.png')
    plt.close('all')
    
test_floodings.round(3).to_csv('./test/figs/floodings.csv')
