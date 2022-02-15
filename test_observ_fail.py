# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:53:07 2022

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


fail_dir = './test/observ_fail/'
inps = [inp for inp in listdir(fail_dir) if inp.endswith('.inp') and 'DQN' in inp]
times = [(read_inp_file(fail_dir+inp).OPTIONS.START_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.START_TIME,
          read_inp_file(fail_dir+inp).OPTIONS.END_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.END_TIME) for inp in inps]
times = [(tu[0].strftime('%m/%d/%Y')+' '+tu[1].strftime('%H:%M:%S'),
          tu[2].strftime('%m/%d/%Y')+' '+tu[3].strftime('%H:%M:%S')) for tu in times]

columns = ['BC','DQN','VDN']
cen_datas = []
vdn_datas = []
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
columns = ['BC','DQN','VDN']
fig = plt.figure(figsize=(30,5),dpi = 600)

for i,data in enumerate(cen_datas):
    refer = refers[i]
    ax = fig.add_subplot(1,5,i+1)
    labels = [str(i*0.1)[:3] for i in range(1,11)]
    data = data.T
    cen_stats = cbook.boxplot_stats(data, bootstrap=10000)
    vdn_stats = cbook.boxplot_stats(vdn_datas[i].T,bootstrap=10000)

    cen = ax.bxp(cen_stats,patch_artist=True,showfliers=False,positions=(0.7,1.7,2.7,3.7,4.7,5.7,6.7,7.7,8.7,9.7),widths=0.3)
    for patch in cen['boxes']:
        patch.set_facecolor('salmon')
    for med in cen['medians']:
        med.set_color('black')
    vdn = ax.bxp(vdn_stats,patch_artist=True,showfliers=False,positions=(1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1),widths=0.3)
    for patch in vdn['boxes']:
        patch.set_facecolor('violet')   
    for med in vdn['medians']:
        med.set_color('black')
    lines = []
    colors = ['orange','red','green']
    for j,ref in enumerate(refer):
        line = ax.axhline(y=ref,linestyle='--',label=columns[j],color=colors[j])
        lines.append(line)
    ax.set_title(times[i][0][:10])
    ax.legend(lines+[cen['boxes'][0],vdn['boxes'][0]],[l.get_label() for l in lines]+['DQN_fail','VDN_fail'],loc='upper left')
    plt.xticks([1,2,3,4,5,6,7,8,9,10],labels)
    ax.set_xlabel('Failure probability')
    ax.set_ylabel('Accumulated CSO volume ($\mathregular{10^3} \mathregular{m^3}$)')
fig.savefig(fail_dir+'fail.png',dpi=600)



# datas = np.load(fail_dir+'fail_data.npy',allow_pickle=True)
# refers = np.load(fail_dir+'fail_refer.npy')
# columns = ['BC','DQN','VDN']
# fig = plt.figure(figsize=(25,5),dpi = 600)
# for i,data in enumerate(datas):
#     refer = refers[i]
#     ax = fig.add_subplot(1,5,i+1)
#     labels = list('01234')
#     mins = [refer[-2]]+[min(dat) for dat in data]
#     means = [refer[-2]]+[sum(dat)/len(dat) for dat in data]
#     maxs = [refer[-2]]+[max(dat) for dat in data]
#     ax.scatter(0,refer[-2],marker='*',c='r')
#     for j,dat in enumerate(data):
#         ax.scatter([j+1 for _ in range(len(dat))],dat,marker='*',c='r')
#     minss = ax.plot(labels,mins,'--',c='r',label='min')
#     maxss = ax.plot(labels,maxs,'--',c='r',label='max')
#     meanss = ax.plot(labels,means,'r',label='mean',lw=2)
#     lines = minss+maxss+meanss

#     bc = ax.axhline(y=refer[0],linestyle='--',c='green',label=columns[0])
#     # dqn = ax.axhline(y=refer[1],linestyle='--',c='red',label=columns[1])
#     vdn = ax.axhline(y=refer[2],linestyle='--',c='blue',label=columns[2])
#     lines.extend([bc,vdn])
#     ax.legend(lines,[l.get_label() for l in lines],loc='upper left')
#     datestr = times[i][0][:10]
#     ax.set_title(datestr)
#     ax.set_xlabel('Failure level')
#     ax.set_ylabel('Accumulated CSO volume ($\mathregular{10^3} \mathregular{m^3}$)')
# fig.savefig(fail_dir+'fail2.png',dpi=600)