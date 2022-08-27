# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 19:26:49 2021

@author: MOMO
"""

import random
from collections import deque
import numpy as np
import os

class RandomMemory():
    def __init__(self,limit,cwd=None,load=False):
        self.items = ['state','action','reward','next_state','done']
        self.limit = limit
        self.cwd = cwd
        self.cur_capa = 0
        if load:
            self.load(cwd)
            self.cur_capa = len(self.reward)
        else:
            for item in self.items:
                setattr(self,item,deque(maxlen=limit))
            self.cur_capa = 0

    def __len__(self):
        return self.cur_capa
        # return len(self.experiences)
    
    def sample(self, batch_size):
        indices = random.sample(range(self.cur_capa),batch_size)
        batch = [[getattr(self,item)[ind] for ind in indices] for item in self.items]
        return batch
        
    def update(self, trajs):
        for traj in trajs:
            for idx,item in enumerate(self.items):
                getattr(self,item).append(traj[idx])
            # self.experiences.append(tuple(traj))
        self.cur_capa = min(self.limit,self.cur_capa + len(trajs))

    def save(self,cwd=None):
        cwd = self.cwd if cwd is None else cwd
        for item in self.items:
            data = np.array(getattr(self,item))
            np.save(os.path.join(cwd,'experience_%s.npy'%item),data)
            print('Save experience %s'%item)


    def load(self,cwd=None):
        cwd = self.cwd if cwd is None else cwd
        for item in self.items:
            data = np.load(os.path.join(cwd,'experience_%s.npy'%item)).tolist()
            setattr(self,item,deque(data,maxlen=self.limit))
            print('Load experience %s'%item)

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
