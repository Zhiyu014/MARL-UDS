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


