# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:01:36 2021

@author: MOMO
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family = 'Times New Roman')

# Comparison between VDN & IQL
iql_train = np.load('./model/IQL/episode_reward_history.npy')
iql_test = np.load('./model/IQL/test_reward_history.npy')

vdn_train = np.load('./model/VDN/episode_reward_history.npy')
vdn_test = np.load('./model/VDN/test_reward_history.npy')

fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,5))
iql = ax.plot(np.arange(len(iql_test)),iql_test)
ax.set_xlabel('Episode')
ax.set_ylabel('Testing Score')
vdn = ax.plot(np.arange(len(vdn_test)),vdn_test)
ax.legend(iql+vdn,['IQL','VDN'],loc='lower right')
plt.savefig('./IQLvsVDN_test.png',dpi=600)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize = (10,5))
iql = ax2.plot(np.arange(len(iql_train)),iql_train)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Training Reward')
vdn = ax2.plot(np.arange(len(vdn_train)),vdn_train)
ax2.legend(iql+vdn,['IQL','VDN'],loc='lower right')
plt.savefig('./IQLvsVDN_train.png',dpi=600)

# Comparison of different reward functions
from os import listdir
dirs = listdir()
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,5))
fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize = (10,5))

lines1 = []
lines2 = []
for di in [dirs[1],dirs[0],dirs[-1]]:
    train = np.load('./'+di+'/episode_reward_history.npy').tolist()
    train.pop(5000)
    train_norm = [(va-min(train))/(max(train)-min(train)) for va in train]
    line1 = ax2.plot(np.arange(len(train_norm)),train_norm,label = di)
    lines1+=line1

    
    test = np.load('./'+di+'/test_reward_history.npy').tolist()
    line2 = ax.plot(np.arange(len(test)),test,label = di)
    lines2+=line2
    
ax.set_xlabel('Episode')
ax.set_ylabel('Testing Score')    
ax2.set_xlabel('Episode')
ax2.set_ylabel('Normalized Training Reward')
ax.legend([lines2[1],lines2[0],lines2[-1]],dirs,loc='lower right')
ax2.legend([lines1[1],lines1[0],lines1[-1]],dirs,loc='lower right')
fig.savefig('./reward_testing.png',dpi=600)
fig2.savefig('./reward_training.png',dpi=600)

