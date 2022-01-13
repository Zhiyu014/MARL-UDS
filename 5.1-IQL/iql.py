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


class IQL:
    def __init__(self,agents = None,memory = None, gamma=0.9, batch_size=47,
            loss_fn = ks.losses.MeanSquaredError(),
            optimizer = ks.optimizers.Adam(),
            agent_action_num = 5,
            update_interval = 5,
            adj_matrix = None):
        self.agents = agents
        self.n_agents = len(agents)
        self.adj_matrix = adj_matrix if adj_matrix is not None else eye(self.n_agents,dtype = float64)
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.step = 0
        self.train_interval = 1
        self.update_interval = update_interval
        self.episode = 0
        self.agent_action_num = agent_action_num

        # self.trainable_variables += self.model.trainable_variables
        # self.target_trainable_variables += self.target_model.trainable_variables
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    
    def update_memory(self, o,o_,s,s_,r,a):
        self.memory.append(s,o,a,r,s_,o_)



        
    def _experience_replay(self,batch_size):
        self.batch_size = batch_size
        _, o, a, r, _, o_ = self.memory.sample(self.batch_size)
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        loss = []

        for idx,agent in enumerate(self.agents):
            actions = agent.model(o_[:,idx,:])
            argmax_actions = ks.backend.argmax(actions)
            # target_masks.append(argmax_actions)
            acts = ks.backend.argmax(a[:,idx,:])
            target_q_value = reduce_sum(agent.target_model(o_[:,idx,:])*one_hot(argmax_actions,self.agent_action_num),axis=1)
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
            y_pred = reduce_sum(model(o)*one_hot(acts,self.agent_action_num))
            loss_value = self.loss_fn(target, y_pred)
        grads = tape.gradient(loss_value, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value.numpy()

    def _test_loss(self,o,o_,s,s_,r,a):
        _, o, a, r, _, o_ = self.memory.sample(self.batch_size)
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        loss = []

        for idx,agent in enumerate(self.agents):
            actions = agent.model(o_[:,idx,:])
            argmax_actions = ks.backend.argmax(actions)
            # target_masks.append(argmax_actions)
            acts = ks.backend.argmax(a[:,idx,:])
            target_q_value = reduce_sum(agent.target_model(o_[:,idx,:])*one_hot(argmax_actions,self.agent_action_num),axis=1)
            target = r + self.gamma * target_q_value
            y_pred = reduce_sum(agent.model(o[:,idx,:])*one_hot(acts,self.agent_action_num))
            los = self.loss_fn(target, y_pred)
            loss.append(los.numpy())   
        return loss

    def _hard_update_target_model(self):
        """ for hard update """
        if self.episode%self.update_interval == 0:
            for agent in self.agents:
                agent._hard_update_target_model()

    def _soft_update_target_model(self):
        for agent in self.agents:
            agent._soft_update_target_model()

    def save(self):
        for i,agent in enumerate(self.agents):
            agent.save(i)
            
    def load(self):
        for i,agent in enumerate(self.agents):
            agent.load(i)