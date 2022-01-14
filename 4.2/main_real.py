# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:58:18 2021

@author: MOMO
"""
from env_swmm_real import Ast
from vdn import VDN
from qagent import QAgent
from memory import RandomMemory
from tensorflow.keras import losses,optimizers
from swmm_api import read_inp_file
import matplotlib.pyplot as plt
from numpy import arange
import numpy as np
from datetime import datetime,timedelta
from os.path import exists
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999	


# init SWMM environment and agents
env = Ast()
agents = []
for i in range(env.n_agents):
    agent = QAgent(epsilon_decay,env.action_size,env.observ_size,dueling=True,epsilon=1)
    agents.append(agent)
    
    
update_num = env.update_num
basic_batch_size = 256
basic_update_times = 10

memory = RandomMemory(limit=700000, agent_num=4,update_num = update_num)

loss_fn = losses.MeanSquaredError()
optimizer = optimizers.Adam()
vdnn = VDN(
    agents=agents,
    memory=memory,
    batch_size=basic_batch_size,
    optimizer=optimizer)
# vdnn.load()

N = 5000
rain_num = 20
#times = env.rand_events(rain_num,20,60,7)
times = np.load('./train/times.npy')
# env.generate_file(times)

test_time = ('08/22/2006 02:50:00','08/22/2006 11:45:00')
bc_score = env.test_bc(test_time)
efd_score = env.test_efd(test_time)
#
#train_loss_history = np.load('./model/train_loss_history.npy').tolist()
#test_loss_history = np.load('./model/test_loss_history.npy').tolist()
#episode_reward_history = np.load('./model/episode_reward_history.npy').tolist()
#test_reward_history = np.load('./model/test_reward_history.npy').tolist()
train_loss_history = []
test_loss_history = []
episode_reward_history = []
test_reward_history = []


for n in range(N):
    rewards = []
    for num,time in enumerate(times):
        print('Sampling times:  %s'%n)
        env.filedir = env.inp_train_file.format(num)        
        observs,next_observs,reward,actions,_ = env.run_simulation(vdnn.agents)
        vdnn.update_memory(observs,next_observs,reward,actions,env.update_num)
        print('Reward on Rain %s:   '%num + str(sum(reward)))
        rewards.append(sum(reward))
    print('Sampling complete')
    print('Upgrading')
    k = 1 + len(memory) / memory.limit   
    batch_size = int(k * basic_batch_size)  # change the batch size gradually
    update_times = int(k * basic_update_times)      # change the update times gradually    
    if n>1:
        for _ in range(update_times):
            train_loss = vdnn._experience_replay(batch_size)
            train_loss_history.append(train_loss)
    print('Upgrade complete')
    for agent in vdnn.agents:
        agent._epsilon_update()
    episode_reward = sum(rewards)
    episode_reward_history.append(episode_reward)

    test_reward,_,test_loss = env.test(vdnn,test_time)
    test_reward_history.append(test_reward)
    test_loss_history.append(test_loss)
    
    print('Baseline Reward:   %s'%bc_score)
    print('EFD Reward:   %s'%efd_score)
    
vdnn.save()
np.save('./model/train_loss_history.npy',np.array(train_loss_history))
np.save('./model/test_loss_history.npy',np.array(test_loss_history))
np.save('./model/episode_reward_history.npy',np.array(episode_reward_history))
np.save('./model/test_reward_history.npy',np.array(test_reward_history))


fig,((axL,axP),(axM,axR)) = plt.subplots(nrows=2,ncols=2,figsize = (10,10), dpi=1200)
axL.plot(arange(len(episode_reward_history)),episode_reward_history,label = 'reward')
axL.set_xlabel('episode')
axL.set_title('episode reward history')
axL.legend(loc='lower right')

axP.plot(arange(len(train_loss_history)),train_loss_history,label='loss')
axP.set_xlabel('episode')
axP.set_title("vdn's training loss")
axP.legend(loc='upper right')


axM.plot(arange(len(test_reward_history)),test_reward_history,label = 'reward')
axM.set_xlabel('episode')
axM.set_title('test score history')
axM.legend(loc='lower right')

axR.plot(arange(len(test_loss_history)),test_loss_history,label='loss')
axR.set_xlabel('episode')
axR.set_title("vdn's test loss")
axR.legend(loc='upper right')


plt.savefig('results.png')




