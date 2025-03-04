# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,reduce_max,float32,reduce_sum
from tensorflow import keras as ks
from tensorflow import expand_dims,matmul
from .qagent import QAgent
from numpy import argmax,array,save,load
from os.path import join
import random

class IQL:
    def __init__(self,
            observ_space: list,
            action_shape: list,
            args = None,
            act_only = False):

        self.name = "IQL"
        self.model_dir = args.cwd

        self.if_recurrent = getattr(args,"if_recurrent",False)
        self.n_agents = getattr(args, "n_agents", 2)

        self.seq_len = getattr(args,"seq_len",3) if self.if_recurrent else None
        self.agents = [QAgent(action_shape[i],len(observ_space[i]),args,self.seq_len)
                    for i in range(self.n_agents)]
                    
        self.observ_space = observ_space
        self.action_shape = action_shape
        self.action_table = getattr(args,'action_table',None)
        self.if_norm = getattr(args,'if_norm',False)
        if self.if_norm:
            self.state_norm = array([[i for _ in range(args.state_shape)] for i in range(2)])
        self.epsilon = getattr(args,'epsilon',1)

        if not act_only:
            self.double = getattr(args,"if_double",True)
            self.gamma = getattr(args, "gamma", 0.98)
            self.batch_size = getattr(args,"batch_size",256)
            self.learning_rate = getattr(args,"learning_rate",1e-5)
            self.repeat_times = getattr(args,"repeat_times",2)
            self.global_reward = getattr(args,'global_reward',True)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.target_update_func = self._hard_update_target_model if self.update_interval >1\
                else self._soft_update_target_model

            self.episode = getattr(args,'episode',0)
            # self.reward_norm = (0,1)

            self.trainable_variables = []
            self.target_trainable_variables = []
            for agent in self.agents:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

            self.loss_fn = ks.losses.get(args.loss_function)
            self.optimizer = ks.optimizers.get(args.optimizer)
            self.optimizer.learning_rate = self.learning_rate



        if args.if_load:
            self.load()
            # print("Load network: "+args.cwd)

    def act(self,state,train=True):
        if train and random.random() < self.epsilon:
            action = [random.randint(0,self.action_shape[i]-1) for i in range(self.n_agents)]
        else:
            if self.if_recurrent:
                # Normalize the state
                if self.if_norm:
                    state = [self._normalize_state(obs) for obs in state]
                    state =  [state[0] for _ in range(self.seq_len-len(state))]+state \
                        if len(state)<self.seq_len else state
            else:
                if self.if_norm:
                    state = self._normalize_state(state)
            # Split state into multiple observations
            observ = self._split_observ([state])
            # Get action from Q table
            action = [argmax(agent.act(observ[i])) for i,agent in enumerate(self.agents)]
        return action

    def convert_action_to_setting(self,action):
        if self.action_table is not None:
            setting = self.action_table[tuple(action)]
            return setting
        else:
            setting = [int(act) for act in action]
            return setting

    def update_net(self,memory,batch_size=None):
        # Update the state & reward normalization paras
        if self.if_norm:
            self.state_norm = memory.get_state_norm()
        # self.reward_norm = memory.get_reward_norm()

        batch_size = self.batch_size if batch_size is None else batch_size
        update_times = int(1 + 4 * len(memory) / memory.limit) * self.repeat_times
        losses = []
        for _ in range(update_times):
            s, a, r, s_, d = memory.sample(batch_size)
            if self.if_norm:
                s,s_ = self._normalize_state(s),self._normalize_state(s_)
            o,o_ = self._split_observ(s),self._split_observ(s_)
            loss = self._experience_replay(o, a, r, o_, d)
            self.target_update_func()
            losses.append(loss)
        return losses

    def evaluate_net(self,trajs):
        s, a, r, s_, d = [[traj[i] for traj in trajs] for i in range(5)]
        if self.if_norm:
            s,s_ = self._normalize_state(s),self._normalize_state(s_)

        if self.if_recurrent:
            s = [[s[0] for _ in range(self.seq_len-i-1)]+s[:i+1] for i in range(self.seq_len-1)]+\
                [s[i:i+self.seq_len] for i in range(len(s)-self.seq_len+1)]
            s_ = [[s_[0] for _ in range(self.seq_len-i-1)]+s_[:i+1] for i in range(self.seq_len-1)]+\
                [s_[i:i+self.seq_len] for i in range(len(s_)-self.seq_len+1)]      

        o,o_ = self._split_observ(s),self._split_observ(s_)
        loss = self._test_loss(o ,a ,r ,o_ ,d)
        return loss
        

    def _normalize_state(self,s):
        # Normalize the state & reward
        s = ((array(s)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        # r = ((array(r)-self.reward_norm[0])/self.reward_norm[1]).tolist()
        return s

    def _split_observ(self,s):
        # Split as multi-agent & convert to tensor
        if self.if_recurrent:
            o = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                   for sis in si] for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        else:
            o = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                   for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        return o


    def _experience_replay(self, o, a, r, o_, d):
        o,o_ = [[convert_to_tensor(oi,dtype=float32) for oi in x] for x in [o,o_]]
        r,d = [convert_to_tensor(i,dtype=float32) for i in [r,d]]
        a = convert_to_tensor(a)

        loss = []
        for idx,agent in enumerate(self.agents):
            if self.double:
                argmax_actions = ks.backend.argmax(agent.forward(o_[idx]))
                target_q_value = reduce_sum(agent.forward(o_[idx],target=True)*one_hot(argmax_actions,self.action_shape[idx]),axis=1)
            else:
                target_q_value = reduce_max(agent.forward(o_[idx],target=True),axis=1)
            if self.global_reward:
                target = r + self.gamma * target_q_value * (1-d)
            else:
                target = r[:,idx] + self.gamma * target_q_value * (1-d)

            # los = self._train_on_agent(agent.model,o[idx],a[:,idx],target)
            with GradientTape() as tape:
                tape.watch(agent.model.trainable_variables)
                y_pred = reduce_sum(agent.forward(o[idx])*one_hot(a[:,idx],self.action_shape[idx]),axis=1)
                loss_value = self.loss_fn(target, y_pred)
            grads = tape.gradient(loss_value, agent.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
            loss.append(loss_value.numpy())
        
        return loss

    def _test_loss(self, o, a, r, o_, d):
        o,o_ = [[convert_to_tensor(oi,dtype=float32) for oi in x] for x in [o,o_]]
        r,d = [convert_to_tensor(i,dtype=float32) for i in [r,d]]
        a = convert_to_tensor(a)

        loss = []
        for idx,agent in enumerate(self.agents):
            if self.double:
                argmax_actions = ks.backend.argmax(agent.forward(o_[idx]))
                target_q_value = reduce_sum(agent.forward(o_[idx],target=True)*one_hot(argmax_actions,self.action_shape[idx]),axis=1)
            else:
                target_q_value = reduce_max(agent.forward(o_[idx],target=True),axis=1)
            if self.global_reward:
                target = r + self.gamma * target_q_value * (1-d)
            else:
                target = r[:,idx] + self.gamma * target_q_value * (1-d)
            y_pred = reduce_sum(agent.forward(o[idx])*one_hot(a[:,idx],self.action_shape[idx]),axis=1)
            los = self.loss_fn(target, y_pred)
            loss.append(los.numpy())   
        return loss
            
    def episode_update(self,episode,epsilon):
        self.episode = episode
        self.epsilon = epsilon

    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            for agent in self.agents:
                agent._hard_update_target_model()

    def _soft_update_target_model(self):
        for agent in self.agents:
            agent._soft_update_target_model(self.update_interval)


    def save(self,model_dir=None,norm=True,agents=True):
        # Save the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                save(join(self.model_dir,'state_norm.npy'),self.state_norm)
            else:
                save(join(model_dir,'state_norm.npy'),self.state_norm)

        # Save the agent paras
        if agents:
            for i,agent in enumerate(self.agents):
                agent.save(i,model_dir)
            
    def load(self,model_dir=None,norm=True,agents=True):
        # Load the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                self.state_norm = load(join(self.model_dir,'state_norm.npy'))
            else:
                self.state_norm = load(join(model_dir,'state_norm.npy'))

        # Load the agent paras
        if agents:
            for i,agent in enumerate(self.agents):
                agent.load(i,model_dir)