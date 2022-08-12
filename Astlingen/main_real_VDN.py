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
# import tqdm
import matplotlib.pyplot as plt
from numpy import arange
import numpy as np
from datetime import datetime,timedelta
from os.path import exists
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999	


# init SWMM environment and agents
env = Ast()    
    
update_num = env.update_num
basic_batch_size = 512
basic_update_times = 10

memory = RandomMemory(limit=700000, agent_num=4,update_num = update_num)

loss_fn = losses.MeanSquaredError()
optimizer = optimizers.Adam()

vdnn = VDN(
    memory=memory,
    action_size=env.action_size,
    RGs = [2,0,2,1],
    batch_size=basic_batch_size,
    optimizer=optimizer,
    model_dir='./model/VDN/')
# vdnn.load()
N = 5000
rain_num = 50
try:
    times = np.load('./train/train_data.npy')
except:
    times = env.rand_events(rain_num,20,60,7)
env.generate_file(times,name='VDN')

test_time = ('08/22/2006 02:50:00','08/22/2006 11:45:00')
bc_score = env.test_bc(test_time)

train_loss_history = []
test_loss_history = []
episode_reward_history = []
test_reward_history = []


for n in range(N):
    rewards = []
    for num,time in enumerate(times):
        print('Sampling times:  %s'%n)
        env.filedir = env.inp_train_file.format(vdnn.name,num)
        observs,next_observs,reward,actions,_ = env.run_simulation(vdnn)
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
    vdnn._epsilon_update()
    episode_reward = sum(rewards)
    episode_reward_history.append(episode_reward)

    test_reward,acts,test_loss = env.test(vdnn,test_time)
    test_reward_history.append(test_reward)
    test_loss_history.append(test_loss)
    
    print('Baseline Reward:   %s'%bc_score)
    
vdnn.save()
np.save(vdnn.model_dir+'train_loss_history.npy',np.array(train_loss_history))
np.save(vdnn.model_dir+'test_loss_history.npy',np.array(test_loss_history))
np.save(vdnn.model_dir+'episode_reward_history.npy',np.array(episode_reward_history))
np.save(vdnn.model_dir+'test_reward_history.npy',np.array(test_reward_history))





fig,((axL,axP),(axM,axR)) = plt.subplots(nrows=2,ncols=2,figsize = (10,10), dpi=1200)
axL.plot(arange(len(episode_reward_history)),episode_reward_history,label = 'reward')
axL.set_xlabel('episode')
axL.set_title('episode reward history')
axL.legend(loc='lower right')

# for idx in range(vdnn.n_agents):
#     axP.plot(arange(len(train_loss_history)),[loss[idx] for loss in train_loss_history],label='Agent %s'%idx)
axP.plot(arange(len(train_loss_history)),train_loss_history,label='loss')
axP.set_xlabel('episode')
axP.set_title("training loss")
axP.legend(loc='upper right')


axM.plot(arange(len(test_reward_history)),test_reward_history,label = 'reward')
axM.set_xlabel('episode')
axM.set_title('test score history')
axM.legend(loc='lower right')

# for idx in range(vdnn.n_agents):
#     axR.plot(arange(len(test_loss_history)),[loss[idx] for loss in test_loss_history],label='Agent %s'%idx)
axR.plot(arange(len(test_loss_history)),test_loss_history,label='loss')
axR.set_xlabel('episode')
axR.set_title("test loss")
axR.legend(loc='upper right')


plt.savefig('VDN.png')




