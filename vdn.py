# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,transpose,cast,float64,reduce_sum,reduce_max,eye
from tensorflow import keras as ks
from tensorflow.keras.activations import sigmoid,elu,relu,linear
from tensorflow.keras.layers import LayerNormalization,ReLU,BatchNormalization
from tensorflow import expand_dims,matmul
from numpy import argmax
from qagent import QAgent

class VDN:
    def __init__(self,agents = None,memory = None, gamma=0.98, batch_size=47,
            loss_fn = ks.losses.MeanSquaredError(),
            optimizer = ks.optimizers.Adam(),
            observ_size = 4,
            action_size = 5,
            n_agents = 4,
            RGs = None,
            epsilon_decay=0.999,
            epsilon=1,
            epsilon_min=0.1,
            update_interval = 5,
            model_dir='./model/'):
        # self.RGs = None
        self.RGs = RGs #TODO:split rgs
        if agents is None:
            self.agents = [QAgent(action_size,observ_size,n_agents,
                            embed_size=128,dueling=True,
                            epsilon_decay=epsilon_decay,epsilon=epsilon,epsilon_min=epsilon_min,
                            model_dir=model_dir)
                           for i in range(n_agents)]
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

        models = []
        target_models = []
        self.trainable_variables = None
        self.target_trainable_variables = None
        for agent in self.agents:
            models.append(agent.model)
            target_models.append(agent.target_model)
            if self.trainable_variables is None:
                self.trainable_variables = agent.model.trainable_variables
                self.target_trainable_variables = agent.target_model.trainable_variables
            else:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

        # self.model = MixingNet(models)
        # self.target_model = MixingNet(target_models)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.name = 'VDN'
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
        
        # target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
        #                  for idx,agent in enumerate(self.agents)]
        # target_masks = convert_to_tensor(target_masks)

        target_q_values = [reduce_max(agent.target_model(o_[:,idx,:]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        target_q_tot = reduce_sum(convert_to_tensor(target_q_values),axis=0)
        
        r = expand_dims(r, 1)
        
        # target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_tot
        # discounted_reward_batch = cast(discounted_reward_batch,float64)
        targets = r + discounted_reward_batch        
        
        # masks = transpose(a)
        with GradientTape() as tape:
            tape.watch(o)
            q_values = [reduce_sum(agent.model(o[:,idx,:])*one_hot(a[:,idx],self.action_size),axis=1)
                    for idx,agent in enumerate(self.agents)]
            q_tot = reduce_sum(convert_to_tensor(q_values),axis=0)
            loss_value = self.loss_fn(targets, q_tot)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # loss = self._train_on_batch(o,masks,targets)
        
        if self.update_interval > 1:
            self._hard_update_target_model()
        else:
            self._soft_update_target_model()
        self.episode += 1
        return loss_value.numpy()

    def _train_on_agent(self,o,masks,target,idx):
        with GradientTape() as tape:
            tape.watch(o)
            y_preds = self.model(o,masks)
            loss_value = self.loss_fn(target, y_preds[:,idx])
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value.numpy()

    def _train_on_batch(self, o, masks, targets):
        
        with GradientTape() as tape:
            tape.watch(o)
            y_preds = self.model(o, masks)
            loss_value = self.loss_fn(targets, y_preds)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
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

        # target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
        #                  for idx,agent in enumerate(self.agents)]
        # target_masks = convert_to_tensor(target_masks)

        target_q_values = [reduce_max(agent.target_model(o_[:,idx,:]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        target_q_tot = reduce_sum(convert_to_tensor(target_q_values),axis=0)
        
        r = expand_dims(r, 1)
        
        # target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_tot
        # discounted_reward_batch = cast(discounted_reward_batch,float64)
        targets = r + discounted_reward_batch        
        
        # masks = transpose(a)
        q_values = [reduce_sum(agent.model(o[:,idx,:])*one_hot(a[:,idx],self.action_size),axis=1)
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

    def _soft_update_target_model(self):
        for agent in self.agents:
            agent._soft_update_target_model()

    def save(self,model_dir=None):
        for i,agent in enumerate(self.agents):
            agent.save(i,model_dir)
            
    def load(self,model_dir=None):
        for i,agent in enumerate(self.agents):
            agent.load(i,model_dir)