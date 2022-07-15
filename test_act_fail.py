# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:46:23 2022

@author: MOMO
"""

from env_swmm_real import Ast
from Cen_RL import Cen_RL
from vdn import VDN
from iql import IQL

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
iqll = IQL(action_size = env.action_size,RGs = [2,0,2,1])
iqll.load(model_dir='./model/IQL/')

fail_dir = './test/act_fail/'
inps = [inp for inp in listdir(fail_dir) if inp.endswith('.inp') and 'DQN' in inp]
times = [(read_inp_file(fail_dir+inp).OPTIONS.START_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.START_TIME,
          read_inp_file(fail_dir+inp).OPTIONS.END_DATE,
          read_inp_file(fail_dir+inp).OPTIONS.END_TIME) for inp in inps]
times = [(tu[0].strftime('%m/%d/%Y')+' '+tu[1].strftime('%H:%M:%S'),
          tu[2].strftime('%m/%d/%Y')+' '+tu[3].strftime('%H:%M:%S')) for tu in times]

datas = []
cen_datas = []
cenbc_datas = []
refers = []
# refers = np.load(fail_dir+'fail_refer.npy')
for idx,time in enumerate(times):
    datestr = time[0][:10]
    env.filedir = fail_dir + inps[idx]
    file = env.filedir
    files = [env.BC_inp,env.filedir.replace('DQN','IQL'),env.filedir,env.filedir.replace('DQN','VDN')]
    test_reward,acts,_ = env.test(iqll,time,filedir=files[1])
    test_reward,acts,_ = env.test(cen_rl,time,filedir=file)
    test_reward,acts,_ = env.test(vdnn,time,filedir=files[-1])
    bc_reward = env.test_bc(time)
    
    refers.append([read_rpt_file(file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr'] 
    for file in files])
    
    # Failure scenario of DQN
    volss = []
    for f in range(1,10):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(cen_rl,time,act_fail=0.1*f,filedir=env.filedir)
            vol = read_rpt_file(env.filedir.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    volss.append([refers[idx][0] for _ in range(50)])
    data = array(volss)
    cen_datas.append(data)
    
    # DQN with BC backup
    volss = []
    for f in range(1,11):
        vols = []
        for j in range(50):
            test_reward,_,_ = env.test(cen_rl,time,inte_fail=0.1*f,f1=cen_rl,f2='BC',filedir=env.filedir)
            vol = read_rpt_file(env.filedir.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    cenbc_datas.append(data)
    
    # DQN with VDN backup
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
    
np.save(fail_dir+'cenbc_fail_data.npy',np.array(cenbc_datas))
np.save(fail_dir+'cen_fail_data.npy',np.array(cen_datas))
np.save(fail_dir+'fail_data.npy',np.array(datas))
np.save(fail_dir+'fail_refer.npy',np.array(refers))
