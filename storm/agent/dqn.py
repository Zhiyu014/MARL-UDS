# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import one_hot,convert_to_tensor,GradientTape,transpose,cast,float32,reduce_max,eye,reduce_sum,squeeze
from tensorflow import keras as ks
from tensorflow import expand_dims,matmul
from numpy import argmax,array,save,load
from .qagent import QAgent
from os.path import join
import random
class DQN:
    def __init__(self,
            observ_space: int,
            action_shape: int or list,
            args = None,
            act_only = False):

        self.name = "DQN"
        self.model_dir = args.cwd

        self.recurrent = getattr(args,"if_recurrent",False)
        self.n_agents = getattr(args, "n_agents", 1)
        self.state_shape = getattr(args, "state_shape", 10)
        self.observ_space = observ_space
        self.action_shape = action_shape
        self.if_mac = getattr(args,'if_mac',False)            
        output_size = sum(self.action_shape) if self.if_mac else self.action_shape
        input_size = self.state_shape if self.if_mac else self.observ_space

        self.seq_len = getattr(args,"seq_len",3) if self.recurrent else None
        self.graph_conv = getattr(args,"global_state",False)
        self.agent = QAgent(output_size,input_size,args,self.seq_len,self.graph_conv) 
        
        self.action_table = getattr(args,'action_table',None)
        self.if_norm = getattr(args,'if_norm',False)
        if self.if_norm:
            self.state_norm = array([[i for _ in range(self.state_shape)] for i in range(2)])
        self.epsilon = getattr(args,'epsilon',1)

        if not act_only:
            self.double = getattr(args,"if_double",True)
            self.gamma = getattr(args, "gamma", 0.98)
            self.batch_size = getattr(args,"batch_size",256)
            self.learning_rate = getattr(args,"learning_rate",1e-5)
            self.repeat_times = getattr(args,"repeat_times",2)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.target_update_func = self._hard_update_target_model if self.update_interval >1\
                else self._soft_update_target_model
            self.episode = getattr(args,'episode',0)
            # self.reward_norm = (0,1)

            self.trainable_variables = []
            self.target_trainable_variables = []
            self.trainable_variables += self.agent.model.trainable_variables
            self.target_trainable_variables += self.agent.target_model.trainable_variables

            self.loss_fn = ks.losses.get(args.loss_function)
            self.optimizer = ks.optimizers.get(args.optimizer)
            self.optimizer.learning_rate = self.learning_rate


        if args.if_load:
            self.load()
            # print("Load network: "+args.cwd)


    def act(self,state,train=True):
        if train and random.random() < self.epsilon:
            # Get random action
            if self.if_mac:
                action = [random.randint(0,shape-1) for shape in self.action_shape]
            else:
                action = (random.randint(0,self.action_shape-1),)
        else:
            if self.recurrent:
                # Normalize the state
                if self.if_norm:
                    state = [self._normalize_state(obs) for obs in state]
                state =  [state[0] for _ in range(self.seq_len-len(state))]+state \
                    if len(state)<self.seq_len else state
            else:
                # Normalize the state
                if self.if_norm:
                    state = self._normalize_state(state)
            # Get action from Q table
            state = expand_dims(convert_to_tensor(state),0)
            a = self.agent.act(state)
            if self.if_mac:
                action = []
                for i,shape in enumerate(self.action_shape):
                    j = sum(self.action_shape[:i])
                    action.append(argmax(a[j:j + shape]))
            else:
                action = (argmax(a),)
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
            loss = self._experience_replay(s,a,r,s_,d)
            self.target_update_func()
            losses.append(loss)
        # deprecated: Decay the exploration epsilon
        # self._epsilon_update()
        return losses


    def evaluate_net(self,trajs):
        s, a, r, s_, d = [[traj[i] for traj in trajs] for i in range(5)]
        if self.if_norm:
            s,s_ = self._normalize_state(s),self._normalize_state(s_)

        if self.recurrent:
            s = [[s[0] for _ in range(self.seq_len-i-1)]+s[:i+1] for i in range(self.seq_len-1)]+\
                [s[i:i+self.seq_len] for i in range(len(s)-self.seq_len+1)]
            s_ = [[s_[0] for _ in range(self.seq_len-i-1)]+s_[:i+1] for i in range(self.seq_len-1)]+\
                [s_[i:i+self.seq_len] for i in range(len(s_)-self.seq_len+1)]      

        loss = self._test_loss(s,a,r,s_,d)
        return loss

    def _normalize_state(self,s):
        # Normalize the state & reward
        s = ((array(s)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        # r = ((array(r)-self.reward_norm[0])/self.reward_norm[1]).tolist()
        return s

    def _experience_replay(self, s, a, r, s_, d):
        
        s,r,s_,d = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d]]
        a = convert_to_tensor(a)
        if not self.if_mac:
            a = squeeze(a)
        
        targets = self._calculate_target(r,s_,d)
        
        loss = self._train_on_batch(s,a,targets)
        return loss


    def _train_on_batch(self, s, a, targets):
        
        with GradientTape() as tape:
            tape.watch(self.trainable_variables)
            y_preds = self.agent.forward(s)
            if self.if_mac:
                y_preds = [reduce_sum(y_preds[:,sum(self.action_shape[:i]):sum(self.action_shape[:i])+shape]*one_hot(a[:,i],shape),axis=1) for i,shape in enumerate(self.action_shape)]
                y_preds = reduce_sum(convert_to_tensor(y_preds),axis=0)
            else:
                y_preds = reduce_sum(y_preds*one_hot(a, depth = self.action_shape),axis=1)
            loss_value = self.loss_fn(targets, y_preds)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value.numpy()

    def _calculate_target(self,r,s_,d):
        if self.double:
            if self.if_mac:
                argmax_actions = [ks.backend.argmax(self.agent.forward(s_)[:,sum(self.action_shape[:i]):sum(self.action_shape[:i])+shape]) for i,shape in enumerate(self.action_shape)]
                target_q_values = self.agent.forward(s_,target=True)
                target_q_values = [reduce_sum(target_q_values[:,sum(self.action_shape[:i]):sum(self.action_shape[:i])+shape]*one_hot(argmax_actions[i],shape),axis=1) for i,shape in enumerate(self.action_shape)]
                target_q_values = reduce_sum(convert_to_tensor(target_q_values),axis=0)
            else:
                argmax_actions = ks.backend.argmax(self.agent.forward(s_))
                target_q_values = reduce_sum(self.agent.forward(s_,target=True)*one_hot(argmax_actions,self.action_shape),axis=1)
        else:
            target_q_values = reduce_max(self.agent.forward(s_,target=True),axis=1) 
        discounted_reward_batch = self.gamma * target_q_values
        targets = r + discounted_reward_batch * (1-d)
        return targets

    def _test_loss(self, s, a, r, s_, d):

        s,r,s_,d = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d]]
        a = convert_to_tensor(a)
        if not self.if_mac:
            a = squeeze(a)
        targets = self._calculate_target(r,s_,d)
        
        y_preds = self.agent.forward(s)
        if self.if_mac:
            y_preds = [reduce_sum(y_preds[:,sum(self.action_shape[:i]):sum(self.action_shape[:i])+shape]*one_hot(a[:,i],shape),axis=1) for i,shape in enumerate(self.action_shape)]
            y_preds = reduce_sum(convert_to_tensor(y_preds),axis=0)
        else:
            y_preds = reduce_sum(y_preds*one_hot(a, depth = self.action_shape),axis=1)
        loss_value = self.loss_fn(targets, y_preds) 
        return loss_value.numpy()

    def episode_update(self,episode,epsilon):
        self.episode = episode
        self.epsilon = epsilon

    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            self.agent._hard_update_target_model()

    def _soft_update_target_model(self):
        self.agent._soft_update_target_model()

    def save(self,model_dir=None,norm=True,agents=True):
        # Save the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                save(join(self.model_dir,'state_norm.npy'),self.state_norm)
            else:
                save(join(model_dir,'state_norm.npy'),self.state_norm)
        # Load the agent paras
        if agents:
            self.agent.save(0,model_dir)
            
    def load(self,model_dir=None,norm=True,agents=True):
        # Load the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                self.state_norm = load(join(self.model_dir,'state_norm.npy'))
            else:
                self.state_norm = load(join(model_dir,'state_norm.npy'))
        # Load the agent paras
        if agents:
            self.agent.load(0,model_dir)
