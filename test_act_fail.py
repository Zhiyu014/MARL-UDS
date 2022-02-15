# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:46:23 2022

@author: MOMO
"""

from env_swmm_real import Ast
from Cen_RL import Cen_RL
from vdn import VDN
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
cen_rl.load(model_dir='./model/DQN/')
vdnn = VDN(action_size = env.action_size,RGs = [2,0,2,1])
vdnn.load(model_dir='./model/VDN/')


fail_dir = './test/act_fail/'
inps = [inp for inp in listdir(fail_dir) if inp.endswith('.inp') and 'DQN' in inp]
times = [(read_inp_file(fail_dir+inp).OPTIONS.START_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.START_TIME,
          read_inp_file(fail_dir+inp).OPTIONS.END_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.END_TIME) for inp in inps]
times = [(tu[0].strftime('%m/%d/%Y')+' '+tu[1].strftime('%H:%M:%S'),
          tu[2].strftime('%m/%d/%Y')+' '+tu[3].strftime('%H:%M:%S')) for tu in times]

columns = ['BC','DQN','VDN']
datas = []
cen_datas = []
refers = []
for idx,time in enumerate(times):
    datestr = time[0][:10]
    env.filedir = fail_dir + inps[idx]
    file = env.filedir
    files = [env.BC_inp,env.filedir,env.filedir.replace('DQN','VDN')]
    test_reward,acts,_ = env.test(cen_rl,time,filedir=file)
    test_reward,acts,_ = env.test(vdnn,time,filedir=files[-1])


    # no_reward = env.test_no(time)
    bc_reward = env.test_bc(time)
    # efd_reward = env.test_efd(time)
    
    refers.append([read_rpt_file(file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr'] 
    for file in files])
    volss = []
    for f in range(1,10):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(cen_rl,time,act_fail=0.1*f,filedir=env.filedir)
            vol = read_rpt_file(env.filedir.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    volss.append([refers[0] for _ in range(50)])
    data = array(volss)
    cen_datas.append(data)
    
    volss = []
    for f in range(1,11):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(cen_rl,time,inte_fail=0.1*f,f1=cen_rl,f2=vdnn,filedir=env.filedir)
            vol = read_rpt_file(env.filedir.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    datas.append(data)
    
np.save(fail_dir+'cen_fail_data.npy',np.array(cen_datas))
np.save(fail_dir+'fail_data.npy',np.array(datas))
# np.save(fail_dir+'fail_refer.npy',np.array(refers))


cen_datas = np.load(fail_dir+'cen_fail_data.npy')
datas = np.load(fail_dir+'fail_data.npy')
refers = np.load(fail_dir+'fail_refer.npy')
columns = ['BC','DQN','VDN']
fig = plt.figure(figsize=(30,5),dpi = 600)

for i,data in enumerate(cen_datas):
    refer = refers[i]
    ax = fig.add_subplot(1,5,i+1)
    labels = [str(i*0.1)[:3] for i in range(1,11)]
    data = data.T
    cen_stats = cbook.boxplot_stats(data, bootstrap=10000)
    stats = cbook.boxplot_stats(datas[i].T, bootstrap=10000)
    
    cen = ax.bxp(cen_stats,patch_artist=True,showfliers=False,positions=(0.7,1.7,2.7,3.7,4.7,5.7,6.7,7.7,8.7,9.7),widths=0.3)    
    for patch in cen['boxes']:
        patch.set_facecolor('salmon')
    for med in cen['medians']:
        med.set_color('black')
    dat = ax.bxp(stats,patch_artist=True,showfliers=False,positions=(1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1),widths=0.3)
    for patch in dat['boxes']:
        patch.set_facecolor('violet')   
    for med in dat['medians']:
        med.set_color('black')
    lines = []
    colors = ['orange','red','green']
    for j,ref in enumerate(refer):
        line = ax.axhline(y=ref,linestyle='--',label=columns[j],color=colors[j])
        lines.append(line)
    ax.set_title(times[i][0][:10])
    ax.legend(lines+[cen['boxes'][0],dat['boxes'][0]],[l.get_label() for l in lines]+['DQN_fail','DQN&VDN'],loc='upper left')
    plt.xticks([1,2,3,4,5,6,7,8,9,10],labels)
    ax.set_xlabel('Failure probability')
    ax.set_ylabel('Accumulated CSO volume ($\mathregular{10^3} \mathregular{m^3}$)')
fig.savefig(fail_dir+'fail.png',dpi=600)
    