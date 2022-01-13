# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 19:26:49 2021

@author: MOMO
"""

import random
from collections import deque, namedtuple

Experience = namedtuple(
    'Experience',
    'state0, observations0, action, reward,observations1, state1')


class Memory():

    def sample(self, **kwargs):
        raise NotImplementedError()

    def append(self, **kwargs):
        raise NotImplementedError()


class RandomMemory(Memory):
    def __init__(self,limit,update_num=47, agent_num=6):
        super(Memory, self).__init__()
        self.update_num = update_num
        self.experiences = deque(maxlen=limit)
        self.agent_num = agent_num
        self.limit = limit

    def __len__(self):
        return len(self.experiences)
    
    def sample(self, batch_size):
        assert batch_size > 1, "batch_size must be positive integer"

        batch_size = min(batch_size, len(self.experiences))
        mini_batch = random.sample(self.experiences, batch_size)
        # mini_batch = [self.experiences[i] for i in batch_num]
        observation_batch = [ba[0] for ba in mini_batch]
        action_batch = [ba[1] for ba in mini_batch]
        reward_batch = [ba[2] for ba in mini_batch]
        next_observation_batch = [ba[3] for ba in mini_batch]
        # for state, observation, action, reward, next_state, next_observation in mini_batch:
        #     state_batch.append(state)
        #     # for i in range(self.agent_num):
        #     observation_batch.append(observation)
        #     next_observation_batch.append(next_observation)
        #     action_batch.append(action)
        #     reward_batch.append(reward)
        #     next_state_batch.append(next_state)


        assert len(action_batch) == batch_size

        return observation_batch, action_batch, reward_batch, next_observation_batch

    def append(self, observation, action, reward,next_observation,update_num):
        self.update_num = update_num
        for i in range(self.update_num):
            self.experiences.append((observation[i], action[i], reward[i], next_observation[i]))