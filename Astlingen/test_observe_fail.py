# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:53:07 2022

@author: MOMO
"""


from env_swmm_real import Ast
from Cen_RL import Cen_RL
from vdn import VDN
from iql import IQL
# import tqdm
from os import listdir
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from swmm_api import read_rpt_file,read_inp_file
from datetime import timedelta,datetime
from numpy import array
import numpy as np
from itertools import combinations
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999


env = Ast(inp_test_file='./test/Real_{0}_{1}.inp')
cen_rl = Cen_RL(action_size = env.action_size)
cen_rl.load(model_dir='./model/DQN/re/')
vdnn = VDN(action_size = env.action_size,RGs = [2,0,2,1])
vdnn.load(model_dir='./model/VDN/re/')
iqll = IQL(action_size = env.action_size,RGs = [2,0,2,1])
iqll.load(model_dir='./model/IQL/0221/re/')


fail_dir = './test/observe_fail/'
inps = [inp for inp in listdir(fail_dir) if inp.endswith('.inp') and 'DQN' in inp]
times = [(read_inp_file(fail_dir+inp).OPTIONS.START_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.START_TIME,
          read_inp_file(fail_dir+inp).OPTIONS.END_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.END_TIME) for inp in inps]
times = [(tu[0].strftime('%m/%d/%Y')+' '+tu[1].strftime('%H:%M:%S'),
          tu[2].strftime('%m/%d/%Y')+' '+tu[3].strftime('%H:%M:%S')) for tu in times]

cen_datas = []
vdn_datas = []
refers = []
for idx,time in enumerate(times):
    datestr = time[0][:10]
    env.filedir = fail_dir + inps[idx]
    file = env.filedir
    files = [env.BC_inp,env.filedir,env.filedir.replace('DQN','IQL'),env.filedir.replace('DQN','VDN')]
    test_reward,acts,_ = env.test(cen_rl,time,filedir=file)
    test_reward,acts,_ = env.test(iqll,time,filedir=files[2])
    test_reward,acts,_ = env.test(vdnn,time,filedir=files[-1])
    bc_reward = env.test_bc(time)
    
    refers.append([read_rpt_file(file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr'] 
    for file in files])
    
    volss = []
    for f in range(1,11):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(cen_rl,time,fail=0.1*f,filedir=env.filedir)
            vol = read_rpt_file(env.filedir.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    cen_datas.append(data)
    
    
    volss = []
    for f in range(1,11):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(vdnn,time,fail=0.1*f,filedir=files[-1])
            vol = read_rpt_file(files[-1].replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    vdn_datas.append(data)
        
np.save(fail_dir+'cen_fail_data.npy',np.array(cen_datas))
np.save(fail_dir+'vdn_fail_data.npy',np.array(vdn_datas))
np.save(fail_dir+'fail_refer.npy',np.array(refers))



# box plot of uncertainty results
cen_datas = np.load(fail_dir+'cen_fail_data.npy')
vdn_datas = np.load(fail_dir+'vdn_fail_data.npy')
refers = np.load(fail_dir+'fail_refer.npy')
times = times[0:1]+times[2:]
columns = ['BC','IQL','DQN','VDN']

fig = plt.figure(figsize=(15,15),dpi = 600)
for i,data in enumerate(cen_datas):
    refer = refers[i]
    ax = fig.add_subplot(2,2,i+1)
    # fig = plt.figure(figsize=(10,10),dpi = 600)
    # ax = fig.add_subplot(1,1,1)
    labels = [str(i*0.1)[:3] for i in range(1,11)]
    data = data.T
    cen_stats = cbook.boxplot_stats(data, bootstrap=10000)
    vdn_stats = cbook.boxplot_stats(vdn_datas[i].T,bootstrap=10000)

    cen = ax.bxp(cen_stats,patch_artist=True,showfliers=False,positions=tuple([i-0.2 for i in range(1,11)]),widths=0.2)
    for patch in cen['boxes']:
        patch.set_facecolor('green')
    for med in cen['medians']:
        med.set_color('black')
    vdn = ax.bxp(vdn_stats,patch_artist=True,showfliers=False,positions=tuple([i+0.2 for i in range(1,11)]),widths=0.2)
    for patch in vdn['boxes']:
        patch.set_facecolor('red')   
    for med in vdn['medians']:
        med.set_color('black')
    lines = []
    colors = ['blue','orange','green','red']
    for j,ref in enumerate(refer):
        line = ax.axhline(y=ref,linestyle='--',label=columns[j],color=colors[j])
        lines.append(line)
    ax.set_title(times[i][0][:10],fontsize=14)
    ax.legend(lines+[cen['boxes'][0],vdn['boxes'][0]],[l.get_label() for l in lines]+['DQN_fail','VDN_fail'],loc='upper left')
    plt.xticks([1,2,3,4,5,6,7,8,9,10],labels)
    ax.set_xlabel('Failure probability',fontsize=14)
    ax.set_ylabel('Accumulated CSO volume ($\mathregular{10^3} \mathregular{m^3}$)',fontsize=14)
    # fig.savefig(fail_dir+'fail%s.png'%times[i][0][:10].replace('/','_'),dpi=600)
fig.savefig(fail_dir+'fail.png',dpi=600)