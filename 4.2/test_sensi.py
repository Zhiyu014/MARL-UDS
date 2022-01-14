# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:04:00 2021

@author: MOMO
"""

from env_swmm_real import Ast
from vdn import VDN
from qagent import QAgent
# import tqdm
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd
from swmm_api import read_rpt_file,read_inp_file
from numpy import array
import numpy as np
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999

env = Ast()


agents = []
for i in range(env.n_agents):
    agent = QAgent(epsilon_decay,env.action_size,env.observ_size,dueling=True)
    agents.append(agent)
vdnn = VDN(agents=agents)
vdnn.load()

# times = env.rand_events(8,30,60,7,train=False)
# test_datestrs = [ti[0][:10] for ti in times]
inps = [inp for inp in listdir('./test/sensi') if inp.endswith('.inp')]
times = [(read_inp_file('./test/sensi/'+inp).OPTIONS.START_DATE,
          read_inp_file('./test/sensi/'+inp).OPTIONS.START_TIME,
          read_inp_file('./test/sensi/'+inp).OPTIONS.END_DATE,
          read_inp_file('./test/sensi/'+inp).OPTIONS.END_TIME) for inp in inps]
times = [(tu[0].strftime('%m/%d/%Y')+' '+tu[1].strftime('%H:%M:%S'),
          tu[2].strftime('%m/%d/%Y')+' '+tu[3].strftime('%H:%M:%S')) for tu in times]


columns = ['Uncontrolled','BC','EFD','VDN']
sensi_table = pd.DataFrame(columns=columns)
datas = []
refers = []
for idx,time in enumerate(times):
    datestr = time[0][:10]
    env.filedir = './test/sensi/' + inps[idx]
    files = [env.do_nothing,env.BC_inp,env.EFD_inp,env.filedir]
    test_reward,acts,_ = env.test(vdnn,time)
    no_reward = env.test_no(time)
    bc_reward = env.test_bc(time)
    efd_reward = env.test_efd(time)
    
    refers.append([read_rpt_file(file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr'] 
    for file in files])
    
    volss = []
    for sen in range(1,6):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(vdnn,time,sensi = 0.1*sen,filedir=files[-1])
            vol = read_rpt_file(files[-1].replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    datas.append(data)

np.save('./test/sensi/sensi_data.npy',np.array(datas))
np.save('./test/sensi/sensi_refer.npy',np.array(refers))

# box plot of uncertainty results
datas = np.load('./test/sensi/sensi_data.npy')
refers = np.load('./test/sensi/sensi_refer.npy')
columns = ['BC','EFD','VDN']
fig = plt.figure(figsize=(25,5),dpi = 600)
for i,data in enumerate(datas):
    refer = refers[i]
    ax = fig.add_subplot(1,5,i+1)
    labels = list('12345')
    data = data.T
    stats = cbook.boxplot_stats(data, labels=labels, bootstrap=10000)
    ax.bxp(stats, showfliers=True)
    lines = []
    colors = ['red','blue','green']
    for j,ref in enumerate(refer[1:]):
        line = ax.axhline(y=ref,linestyle='--',label=columns[j],color=colors[j])
        lines.append(line)
    datestr = times[i][0][:10]
    ax.legend(lines,[l.get_label() for l in lines],loc='upper right')
    ax.set_title(datestr)
    ax.set_xlabel('Uncertainty level')
    ax.set_ylabel('Total flooding volume ($\mathregular{10^3} \mathregular{m^3}$)')
fig.savefig('./test/sensi/sensi.png',dpi=600)
            
            
    