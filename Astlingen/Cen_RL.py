# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,transpose,cast,float64,reduce_max,eye,reduce_sum
from tensorflow import keras as ks
from tensorflow.keras.activations import sigmoid,elu,relu,linear
from tensorflow.keras.layers import LayerNormalization,ReLU,BatchNormalization
from tensorflow import expand_dims,matmul
import random
from numpy import argmax
from qagent import QAgent

class Cen_RL:
    def __init__(self,memory = None, gamma=0.98, batch_size=47,
            loss_fn = ks.losses.MeanSquaredError(),
            optimizer = ks.optimizers.Adam(),
            observ_size = 3,
            action_size = 5,
            n_agents = 4,
            epsilon_decay=0.999,
            epsilon=1,
            epsilon_min=0.1,
            update_interval = 5,
            model_dir='./model/'):
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.step = 0
        self.train_interval = 1
        self.update_interval = update_interval
        self.episode = 0
        self.action_size = action_size
        self.n_agents = n_agents
        
        self.agent = QAgent(action_size,observ_size,n_agents,
                            embed_size=256,input_size = observ_size*n_agents+4,
                            output_size = action_size**n_agents,dueling=True,
                            epsilon_decay=epsilon_decay,epsilon=epsilon,epsilon_min=epsilon_min,
                            model_dir = model_dir)
        
        self.action_size = action_size
        self.q_options = [(i,j,t,k) 
                          for i in range(action_size) 
                          for j in range(action_size) 
                          for t in range(action_size) 
                          for k in range(action_size)]
        self.trainable_variables = None
        self.target_trainable_variables = None

        self.trainable_variables = self.agent.model.trainable_variables
        self.target_trainable_variables = self.agent.target_model.trainable_variables
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.name = 'DQN'
        self.model_dir = model_dir

    
    def update_memory(self, o,o_,r,a,num):
        # a = [tuple([aim*(self.action_size-1) for aim in ai]) for ai in a]
        a = [tuple(ai) for ai in a]
        a = [self.q_options.index(ai) for ai in a]
        self.memory.update_num = num
        self.memory.append(o,a,r,o_)

    def act(self,observ,train):
        a = self.agent.act(observ,train)
        action = self.q_options[argmax(a)]
        # action = [act/(self.action_size-1) for act in action]
        return action
        
    def _experience_replay(self,batch_size):
        self.batch_size = batch_size
        o, a, r, o_ = self.memory.sample(self.batch_size)
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        
        target_q_values = self.agent.target_model(o_)
        discounted_reward_batch = self.gamma * target_q_values
        targets = r + reduce_max(discounted_reward_batch,axis=1) 
        
        loss = self._train_on_batch(o,a,targets)
        
        if self.update_interval > 1:
            self._hard_update_target_model()
        else:
            self._soft_update_target_model()
        self.episode += 1
        return loss


    def _train_on_batch(self, o, a, targets):
        
        with GradientTape() as tape:
            tape.watch(o)
            y_preds = self.agent.model(o)
            y_preds = reduce_sum(y_preds*one_hot(a, depth = self.action_size**self.n_agents),axis=1)
            loss_value = self.loss_fn(targets, y_preds)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value.numpy()

    def _test_loss(self,o,o_,r,a):
        # a = [tuple([aim*(self.action_size-1) for aim in ai]) for ai in a]
        a = [tuple(ai) for ai in a]
        a = [self.q_options.index(ai) for ai in a]
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        
        target_q_values = self.agent.target_model(o_)
        discounted_reward_batch = self.gamma * target_q_values
        targets = r + reduce_max(discounted_reward_batch,axis=1) 
        
        
        y_preds = self.agent.model(o)
        y_preds = reduce_sum(y_preds*one_hot(a, depth = self.action_size**self.n_agents),axis=1)
        loss_value = self.loss_fn(targets, y_preds) 
        loss = loss_value.numpy()
        return loss

    def _epsilon_update(self):
        self.agent._epsilon_update()
            
    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            self.agent._hard_update_target_model()

    def _soft_update_target_model(self):
        self.agent._soft_update_target_model()

    def save(self,model_dir=None):
        self.agent.save(0,model_dir)
            
    def load(self,model_dir=None):
        self.agent.load(0,model_dir)
