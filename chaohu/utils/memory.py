# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 19:26:49 2021

@author: MOMO
"""

import random
from collections import deque
import numpy as np

class RandomMemory():
    def __init__(self,limit):
        self.items = ['state','action','reward','next_state','done']
        for item in self.items:
            setattr(self,item,deque(maxlen=limit))
        # self.experiences = deque(maxlen=limit)
        self.limit = limit
        self.cur_capa = 0

    def __len__(self):
        return self.cur_capa
        # return len(self.experiences)
    
    def sample(self, batch_size):


        # batch_size = min(batch_size, len(self.experiences))
        # mini_batch = random.sample(self.experiences, batch_size)
        # state_batch = [ba[0] for ba in mini_batch]
        # action_batch = [ba[1] for ba in mini_batch]
        # reward_batch = [ba[2] for ba in mini_batch]
        # next_state_batch = [ba[3] for ba in mini_batch]
        # done_batch = [ba[4] for ba in mini_batch]
        indices = random.sample(range(self.cur_capa),batch_size)
        batch = [[getattr(self,item)[ind] for ind in indices] for item in self.items]
        return batch
        
    def update(self, trajs):
        for traj in trajs:
            for idx,item in enumerate(self.items):
                getattr(self,item).append(traj[idx])
            # self.experiences.append(tuple(traj))
        self.cur_capa = min(self.limit,self.cur_capa + len(trajs))


    def get_state_norm(self):
        state = np.asarray(self.state)
        mean = state.mean(axis=0)
        std = state.std(axis=0)
        return np.array([mean,std])


    def get_reward_norm(self):
        reward = np.asarray(self.reward)
        mean = reward.mean()
        std = reward.std()
        return (mean,std)
