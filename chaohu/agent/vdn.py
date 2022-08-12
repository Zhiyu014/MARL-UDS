# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,transpose,cast,float64,reduce_sum,reduce_max,zeros
from tensorflow import keras as ks
from tensorflow.keras.activations import sigmoid,elu,relu,linear
from tensorflow.keras.layers import LayerNormalization,ReLU,BatchNormalization
from tensorflow import expand_dims,matmul
from numpy import argmax,array,save,load
from .qagent import QAgent
from .qragent import QRAgent
from os.path import join

class VDN:
    def __init__(self,
            observ_space: list,
            action_space: list,
            args = None):


        self.recurrent = getattr(args,"if_recurrent",False)
        self.n_agents = getattr(args, "n_agents", 2)
        self.gamma = getattr(args, "gamma", 0.98)
        self.batch_size = getattr(args,"batch_size",256)
        self.learning_rate = getattr(args,"learning_rate",1e-5)
        self.repeat_times = getattr(args,"repeat_times",2)
        self.update_interval = getattr(args,"update_interval",0.005)
        self.target_update_func = self._hard_update_target_model if self.update_interval >1\
             else self._soft_update_target_model

        if self.recurrent:
            self.seq_len = getattr(args,"seq_len",3)
            self.agents = [QRAgent(action_space[i],len(observ_space[i]),self.seq_len,args) 
                           for i in range(self.n_agents)]       
        else:
            self.agents = [QAgent(action_space[i],len(observ_space[i]),args)
                           for i in range(self.n_agents)]
        self.observ_space = observ_space
        self.action_space = action_space
        self.episode = 0
        self.action_table = getattr(args,'action_table')

        self.state_norm = array([[i for _ in range(args.state_shape)] for i in range(2)])
        # self.reward_norm = (0,1)

        self.trainable_variables = []
        self.target_trainable_variables = []
        for agent in self.agents:
            self.trainable_variables += agent.model.trainable_variables
            self.target_trainable_variables += agent.target_model.trainable_variables

        self.loss_fn = ks.losses.get(args.loss_function)
        self.optimizer = ks.optimizers.get(args.optimizer)
        self.optimizer.learning_rate = self.learning_rate

        self.name = getattr(args,"agent_class","VDN")
        self.model_dir = args.cwd

    def act(self,state,train):
        action = []
        for i,agent in enumerate(self.agents):
            if self.recurrent:
                # Normalize the state
                state = [(array(obs)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)
                 for obs in state]
                state =  [state[0] for _ in range(self.seq_len-len(state))]+state \
                    if len(state)<self.seq_len else state
                o = [[obs[idx] for idx in self.observ_space[i]] for obs in state]
            else:
                # Normalize the state
                state = (array(state)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)
                o = [state[idx] for idx in self.observ_space[i]]
            a = agent.act(o,train)
            act = argmax(a)
            action.append(act)
        return action
    
    def convert_action_to_setting(self,action):
        setting = self.action_table[tuple(action)]
        return setting

    def update_net(self,memory):
        # Update the state & reward normalization paras
        self.state_norm = memory.get_state_norm()
        # self.reward_norm = memory.get_reward_norm()

        update_times = int(1 + 4 * len(memory) / memory.limit) * self.repeat_times
        losses = []
        for _ in range(update_times):
            loss = self._experience_replay(memory,self.batch_size)
            self.target_update_func()
            losses.append(loss)
        # Decay the exploration epsilon
        self._epsilon_update()
        return losses

    def evaluate_net(self,trajs):
        s, a, r, s_, d = [[traj[i] for traj in trajs] for i in range(5)]
        loss = self._test_loss(s,a,r,s_,d)
        return loss
        
    def _experience_replay(self,memory,batch_size):
        self.batch_size = batch_size
        s, a, r, s_, d = memory.sample(self.batch_size)
        
        # Normalize the state & reward
        s = ((array(s)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        s_ = ((array(s_)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        # r = ((array(r)-self.reward_norm[0])/self.reward_norm[1]).tolist()

        # Split as multi-agent & convert to tensor
        if self.recurrent:
            o = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                   for sis in si] for si in s]) 
                                   for i in range(self.n_agents)]
            o_ = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                    for sis in si] for si in s_]) 
                                    for i in range(self.n_agents)] 
            s,s_ = [si[-1] for si in s],[si[-1] for si in s_]
        else:
            o = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                   for si in s]) 
                                   for i in range(self.n_agents)]
            o_ = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                    for si in s_]) 
                                    for i in range(self.n_agents)]
        s,r,a,s_,d = [convert_to_tensor(i) for i in [s,r,a,s_,d]]


        
        # target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
        #                  for idx,agent in enumerate(self.agents)]
        # target_masks = convert_to_tensor(target_masks)

        target_q_values = [reduce_max(agent.target_model(o_[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        target_q_tot = reduce_sum(convert_to_tensor(target_q_values),axis=0)
        
        # r = expand_dims(r, 1)
        
        # target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_tot
        # discounted_reward_batch = cast(discounted_reward_batch,float64)
        targets = r + discounted_reward_batch * (1-d)
        
        # masks = transpose(a)
        with GradientTape() as tape:
            tape.watch(o)
            q_values = [reduce_sum(agent.model(o[idx])*one_hot(a[:,idx],self.action_space[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
            q_tot = reduce_sum(convert_to_tensor(q_values),axis=0)
            loss_value = self.loss_fn(targets, q_tot)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # loss = self._train_on_batch(o,masks,targets)
        

        return loss_value.numpy()

    def _test_loss(self,s,a,r,s_,d):
        # a = [[int(aim*(self.action_size-1)) for aim in ai] for ai in a]

        # Normalize the state & reward
        s = ((array(s)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        s_ = ((array(s_)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        # r = ((array(r)-self.reward_norm[0])/self.reward_norm[1]).tolist()


        if self.recurrent:
            o = [[s[0] for _ in range(self.seq_len-i-1)]+s[:i+1] for i in range(self.seq_len-1)]+\
                [s[i:i+self.seq_len] for i in range(len(s)-self.seq_len+1)]
            o_ = [[s_[0] for _ in range(self.seq_len-i-1)]+s_[:i+1] for i in range(self.seq_len-1)]+\
                [s_[i:i+self.seq_len] for i in range(len(s_)-self.seq_len+1)]            
            
            o = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                   for sis in si] for si in o]) 
                                   for i in range(self.n_agents)]
            o_ = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                    for sis in si] for si in o_]) 
                                    for i in range(self.n_agents)] 
        else:        
            o = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                   for si in s]) 
                                   for i in range(self.n_agents)]
            o_ = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                    for si in s_]) 
                                    for i in range(self.n_agents)]
        s,r,a,s_,d = [convert_to_tensor(i) for i in [s,r,a,s_,d]]

        # target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
        #                  for idx,agent in enumerate(self.agents)]
        # target_masks = convert_to_tensor(target_masks)

        target_q_values = [reduce_max(agent.target_model(o_[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        target_q_tot = reduce_sum(convert_to_tensor(target_q_values),axis=0)
        
        # r = expand_dims(r, 1)
        
        # target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_tot
        # discounted_reward_batch = cast(discounted_reward_batch,float64)
        targets = r + discounted_reward_batch * (1-d)
        
        # masks = transpose(a)
        q_values = [reduce_sum(agent.model(o[idx])*one_hot(a[:,idx],self.action_space[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        q_tot = reduce_sum(convert_to_tensor(q_values),axis=0)
        
        # y_preds = self.model(o, masks)
        loss_value = self.loss_fn(targets, q_tot) 
        loss = loss_value.numpy()
        return loss

    def _epsilon_update(self):
        for agent in self.agents:
            agent._epsilon_update()
            
    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            for agent in self.agents:
                agent._hard_update_target_model()
        self.episode += 1

    def _soft_update_target_model(self):
        for agent in self.agents:
            agent._soft_update_target_model(self.update_interval)
        self.episode += 1


    def save(self,model_dir=None,norm=True,agents=True):
        # Save the state normalization paras
        if norm:
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
        if norm:
            if model_dir is None:
                self.state_norm = load(join(self.model_dir,'state_norm.npy'))
            else:
                self.state_norm = load(join(model_dir,'state_norm.npy'))

        # Load the agent paras
        if agents:
            for i,agent in enumerate(self.agents):
                agent.load(i,model_dir)