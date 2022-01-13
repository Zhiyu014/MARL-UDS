# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:01:36 2021

@author: MOMO
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family = 'Times New Roman')

train_path = './model/episode_reward_history.npy'
test_path = './model/test_reward_history.npy'
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,5))


episode_reward_history = np.load(train_path).tolist()
test_reward_history = np.load(test_path).tolist()
train = ax.plot(np.arange(len(episode_reward_history)),episode_reward_history,'orangered')
ax.set_xlabel('Episode')
ax.set_ylabel('Training Reward')

axR = ax.twinx()
axR.yaxis.set_ticks_position('right')
axR.yaxis.set_label_position('right')
test = axR.plot(np.arange(len(test_reward_history)),test_reward_history,'skyblue')
axR.set_xlabel('Episode')
axR.set_ylabel('Testing Score')

axR.legend(train+test,['Training','Testing'],loc='lower right')

plt.savefig('./test/results/train_test.png',dpi=600)
