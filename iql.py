# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021
@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,transpose,cast,float64,reduce_sum,eye
from tensorflow import keras as ks
from tensorflow.keras.activations import sigmoid,elu,relu,linear
from tensorflow.keras.layers import LayerNormalization,ReLU,BatchNormalization
from tensorflow import expand_dims,matmul
from qagent import QAgent
from numpy import argmax

class IQL:
    def __init__(self,agents = None,memory = None, gamma=0.98, batch_size=47,
            loss_fn = ks.losses.MeanSquaredError(),
            optimizer = ks.optimizers.Adam(),
            observ_size = 4,
            action_size = 3,
            n_agents = 4,
            RGs = None,
            epsilon_decay=0.999,
            epsilon=1,
            epsilon_min=0.1,
            update_interval = 5,
            model_dir='./model/'):
        self.RGs = RGs #TODO:split rgs
        if agents is None:
            self.agents = [QAgent(action_size,observ_size,n_agents,
                            embed_size=128,dueling=True,
                            epsilon_decay=epsilon_decay,epsilon=epsilon,epsilon_min=epsilon_min,
                            model_dir=model_dir)
                           for _ in range(n_agents)]
        else:
            self.agents = agents

        self.n_agents = n_agents
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.step = 0
        self.train_interval = 1
        self.update_interval = update_interval
        self.episode = 0
        self.action_size = action_size

        # self.trainable_variables += self.model.trainable_variables
        # self.target_trainable_variables += self.target_model.trainable_variables
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.name = 'IQL'
        self.model_dir = model_dir
    
    def update_memory(self, s,s_,r,a,num):
        if self.RGs is None:
            o = [[si[:4]+[si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)]
                for si in s]
            o_ = [[si[:4]+[si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)]
                for si in s_]
        else:
            o = [[[si[self.RGs[i]]] + [si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)] 
                for si in s]
            o_ = [[[si[self.RGs[i]]] + [si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)] 
                for si in s_]
        # a = [[int(aim*(self.action_size-1)) for aim in ai] for ai in a]
        self.memory.update_num = num
        self.memory.append(o,a,r,o_)

    def act(self,observ,train):
        action = []
        for i,agent in enumerate(self.agents):
            if self.RGs is None:
                o = observ[:4] + [observ[4+i],observ[4+self.n_agents+i],observ[4+self.n_agents*2+i]]
            else:
                o = [observ[self.RGs[i]]] + [observ[4+i],observ[4+self.n_agents+i],observ[4+self.n_agents*2+i]]
            a = agent.act(o,train)
            act = argmax(a)
            action.append(act)
        return action
        
    def _experience_replay(self,batch_size):
        self.batch_size = batch_size
        o, a, r, o_ = self.memory.sample(self.batch_size)
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        loss = []

        for idx,agent in enumerate(self.agents):
            actions = agent.model(o_[:,idx,:])
            argmax_actions = ks.backend.argmax(actions)
            # target_masks.append(argmax_actions)
            acts = a[:,idx]
            target_q_value = reduce_sum(agent.target_model(o_[:,idx,:])*one_hot(argmax_actions,self.action_size),axis=1)
            target = r + self.gamma * target_q_value
            los = self._train_on_agent(agent.model,o[:,idx,:],acts,target)
            loss.append(los)
            # masks.append(acts)
        # loss = self._train_on_batch(o,masks,targets)
        
        if self.update_interval > 1:
            self._hard_update_target_model()
        else:
            self._soft_update_target_model()
        self.episode += 1
        return loss

    def _train_on_agent(self,model,o,acts,target):
        with GradientTape() as tape:
            tape.watch(o)
            y_pred = reduce_sum(model(o)*one_hot(acts,self.action_size),axis=1)
            loss_value = self.loss_fn(target, y_pred)
        grads = tape.gradient(loss_value, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value.numpy()

    def _test_loss(self,s,s_,r,a):
        # a = [[int(aim*(self.action_size-1)) for aim in ai] for ai in a]
        if self.RGs is None:
            o = [[si[:4]+[si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)]
                for si in s]
            o_ = [[si[:4]+[si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)]
                for si in s_]
        else:
            o = [[[si[self.RGs[i]]] + [si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)] 
                for si in s]
            o_ = [[[si[self.RGs[i]]] + [si[4+i],si[4+self.n_agents+i],si[4+self.n_agents*2+i]]
                for i in range(self.n_agents)] 
                for si in s_]
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        loss = []

        for idx,agent in enumerate(self.agents):
            actions = agent.model(o_[:,idx,:])
            argmax_actions = ks.backend.argmax(actions)
            # target_masks.append(argmax_actions)
            acts = a[:,idx]
            target_q_value = reduce_sum(agent.target_model(o_[:,idx,:])*one_hot(argmax_actions,self.action_size),axis=1)
            target = r + self.gamma * target_q_value
            y_pred = reduce_sum(agent.model(o[:,idx,:])*one_hot(acts,self.action_size),axis=1)
            los = self.loss_fn(target, y_pred)
            loss.append(los.numpy())   
        return loss

    def _epsilon_update(self):
        for agent in self.agents:
            agent._epsilon_update()
            
    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            for agent in self.agents:
                agent._hard_update_target_model()

    def _soft_update_target_model(self):
        for agent in self.agents:
            agent._soft_update_target_model()

    def save(self,model_dir=None):
        for i,agent in enumerate(self.agents):
            agent.save(i,model_dir)
            
    def load(self,model_dir=None):
        for i,agent in enumerate(self.agents):
            agent.load(i,model_dir)