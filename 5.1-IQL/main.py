# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:58:18 2021

@author: MOMO
"""
from env_swmm import Ast
from iql import IQL
from qagent import QAgent
from memory import RandomMemory
from tensorflow.keras import losses,optimizers
# import tqdm
from os.path import exists
import matplotlib.pyplot as plt
from numpy import arange,save,array
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999	


# init SWMM environment and agents
env = Ast()
agents = []
for i in range(env.n_agents):
    agent = QAgent(epsilon_decay,env.action_size,env.observ_size,dueling=True)
    agents.append(agent)
    
    
update_num = env.update_num
basic_batch_size = 64
basic_update_times = 5

memory = RandomMemory(limit=250000, agent_num=4,update_num = update_num)

loss_fn = losses.MeanSquaredError()
optimizer = optimizers.Adam()
iqll = IQL(
    agents=agents,
    memory=memory,
    batch_size=basic_batch_size,
    optimizer=optimizer)
# iqll.load()
try:
    with open('./test/rains/train_test.txt','r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split() for line in lines]
        rain = [[line[1],float(line[-1])] for line in lines]
except:
    rain = env.euler_II_Hyeto(r = 0.285,P=4)

N = 8000
rain_num = 20
train_loss_history = []
test_loss_history = []
episode_reward_history = []
test_reward_history = []
episode_reward_mean = 0
loss_mean = 0
bc_reward = env.test_bc(rain)
efd_reward = env.test_efd(rain)
for n in range(N):
    rewards = []
    for num in range(rain_num):
        print('Sampling times:  %s'%n)
        env.train_file = env.train_inp_file%num
        if exists(env.train_file) == False:
            raindata = env.euler_II_Hyeto()
            ts = env.inp.TIMESERIES['ts']
            ts.data = raindata
            env.inp.write_file(env.train_file)
        observs,next_observs,states,next_states,reward,actions,_ = env.run_simulation(iqll.agents)
        iqll.update_memory(observs,next_observs,states,next_states,reward,actions)
        print('Reward on Rain %s:   '%num + str(sum(reward)))
        rewards.append(sum(reward))
    print('Sampling complete')
    print('Upgrading')
    k = 1 + len(memory) / memory.limit   
    batch_size = int(k * basic_batch_size)  # change the batch size gradually
    update_times = int(k * basic_update_times)      # change the update times gradually    
    if n>1:
        for _ in range(update_times):
            train_loss = iqll._experience_replay(batch_size)
            train_loss_history.append(train_loss)
    print('Upgrade complete')
    for agent in iqll.agents:
        agent._epsilon_update()
    episode_reward = sum(rewards)
    episode_reward_history.append(episode_reward)

    test_reward,acts,test_loss = env.test(iqll,rain,test_inp_file = './test/testVDN.inp')
    test_reward_history.append(test_reward)
    test_loss_history.append(test_loss)
    print('Baseline Reward:   %s'%bc_reward)
    print('EFD Reward:   %s'%efd_reward)
iqll.save()
save('./model/train_loss_history.npy',array(train_loss_history))
save('./model/test_loss_history.npy',array(test_loss_history))
save('./model/episode_reward_history.npy',array(episode_reward_history))
save('./model/test_reward_history.npy',array(test_reward_history))


fig,((axL,axP),(axM,axR)) = plt.subplots(nrows=2,ncols=2,figsize = (10,10),dpi=600)
axL.plot(arange(len(episode_reward_history)),episode_reward_history,label = 'reward')
axL.set_xlabel('episode')
axL.set_title('episode reward history')
axL.legend(loc='lower right')

for idx in range(iqll.n_agents):
    axP.plot(arange(len(train_loss_history)),[loss[idx] for loss in train_loss_history],label='Agent %s'%idx)
#axP.plot(arange(len(train_loss_history)),train_loss_history,label='loss')
axP.set_xlabel('episode')
axP.set_title("IQL training loss")
axP.legend(loc='upper right')


axM.plot(arange(len(test_reward_history)),test_reward_history,label = 'reward')
axM.set_xlabel('episode')
axM.set_title('test score history')
axM.legend(loc='lower right')

for idx in range(iqll.n_agents):
    axR.plot(arange(len(test_loss_history)),[loss[idx] for loss in test_loss_history],label='Agent %s'%idx)
#axR.plot(arange(len(test_loss_history)),test_loss_history,label='loss')
axR.set_xlabel('episode')
axR.set_title("IQL testing loss")
axR.legend(loc='upper right')


plt.savefig('iql.png')




