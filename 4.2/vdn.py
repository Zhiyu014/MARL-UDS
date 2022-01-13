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
class SumNet:
    def __init__(self,agents):
        self.agents = agents
        self.n_agents = len(agents)
        
    def __call__(self,observs,masks):
        q_values = []
        # for idx,agent in enumerate(self.agents):
        #     observ, mask = observs[:,idx,:], masks[idx,:]
        #     q_value = agent(observ)
        #     q_value = convert_to_tensor([q_value[i,mask[i]] for i in range(q_value.shape[0])])       # Why mask?  For argmax
        #     q_values.append(q_value)
        q_values = [reduce_sum(agent(observs[:,idx,:])*one_hot(masks[idx,:],agent.output_shape[-1]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        q_values = transpose(convert_to_tensor(q_values,1))
        q_tot = reduce_sum(q_values,axis=1)
        return q_tot

class VDN:
    def __init__(self,agents = None,memory = None, gamma=0.98, batch_size=47,
            loss_fn = ks.losses.MeanSquaredError(),
            optimizer = ks.optimizers.Adam(),
            agent_action_num = 5,
            n_agents = 4,
            update_interval = 5):
        self.agents = agents
        self.n_agents = n_agents
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.step = 0
        self.train_interval = 1
        self.update_interval = update_interval
        self.episode = 0
        self.agent_action_num = agent_action_num

        models = []
        target_models = []
        self.trainable_variables = None
        self.target_trainable_variables = None
        for agent in agents:
            models.append(agent.model)
            target_models.append(agent.target_model)
            if self.trainable_variables is None:
                self.trainable_variables = agent.model.trainable_variables
                self.target_trainable_variables = agent.target_model.trainable_variables
            else:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

        self.model = SumNet(models)
        self.target_model = SumNet(target_models)

        # self.trainable_variables += self.model.trainable_variables
        # self.target_trainable_variables += self.target_model.trainable_variables
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    
    def update_memory(self, o,o_,r,a,update_num):
        self.memory.append(o,a,r,o_,update_num)



        
    def _experience_replay(self,batch_size):
        self.batch_size = batch_size
        o, a, r, o_ = self.memory.sample(self.batch_size)
        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        # masks, target_masks = [], []

        # for idx,agent in enumerate(self.agents):
        #     actions = agent.model(o_[:,idx,:])
        #     argmax_actions = ks.backend.argmax(actions)
        #     # target_mask = one_hot(argmax_actions, depth=self.agent_action_num)
        #     target_masks.append(argmax_actions)
        #     acts = ks.backend.argmax(a[:,idx,:])
        #     # mask = one_hot(acts, depth=self.agent_action_num)
        #     masks.append(acts)

        masks = [ks.backend.argmax(a[:,idx,:]) for idx in range(self.n_agents)]
        target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
                        for idx,agent in enumerate(self.agents)]
        
        masks = convert_to_tensor(masks)
        target_masks = convert_to_tensor(target_masks)


        
        target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_values
        targets = r + discounted_reward_batch     
        
        loss = self._train_on_batch(o,masks,targets)
        
        if self.update_interval > 1:
            self._hard_update_target_model()
        else:
            self._soft_update_target_model()
        self.episode += 1
        return loss

    def _train_on_agent(self,o,masks,target,idx):
        with GradientTape() as tape:
            tape.watch(o)
            y_preds = self.model(o,masks)
            loss_value = self.loss_fn(target, y_preds[:,idx])
        grads = tape.gradient(loss_value, self.trainable_variables)
        # grads = [ele*loss_value.numpy() for ele in grads]
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

    def _test_loss(self,o,o_,r,a):

        o,o_,r,a = convert_to_tensor(o),convert_to_tensor(o_),convert_to_tensor(r),convert_to_tensor(a)
        # masks, target_masks = [], []


        # for idx,agent in enumerate(self.agents):
        #     actions = agent.model(o_[:,idx,:])
        #     argmax_actions = ks.backend.argmax(actions)
        #     # target_mask = one_hot(argmax_actions, depth=self.agent_action_num)
        #     target_masks.append(argmax_actions)
        #     acts = ks.backend.argmax(a[:,idx,:])
        #     # mask = one_hot(acts, depth=self.agent_action_num)
        #     masks.append(acts)

        masks = [ks.backend.argmax(a[:,idx,:]) for idx in range(self.n_agents)]
        target_masks = [ks.backend.argmax(agent.model(o_[:,idx,:])) 
                         for idx,agent in enumerate(self.agents)]

        masks = convert_to_tensor(masks)
        target_masks = convert_to_tensor(target_masks)

        r = expand_dims(r, 1)
        
        target_q_values = self.target_model(o_, target_masks)
        discounted_reward_batch = self.gamma * target_q_values
        # discounted_reward_batch = cast(discounted_reward_batch,float64)
        targets = r + discounted_reward_batch        
        
        
        y_preds = self.model(o, masks)
        loss_value = self.loss_fn(targets, y_preds) 
        loss = loss_value.numpy()
        return loss

    def _hard_update_target_model(self):
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