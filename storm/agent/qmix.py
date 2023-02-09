# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:23:18 2021

@author: MOMO
"""
from tensorflow import convert_to_tensor,GradientTape,transpose,reduce_sum,reduce_max,one_hot,float32
from tensorflow import TensorShape,random as tfrand
from tensorflow import keras as ks
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import elu
from tensorflow import abs as tfabs,reshape,matmul
from numpy import array,argmax,load,save
from .qagent import QAgent
from os.path import join
import random


class MixingNet(ks.Model):
    def __init__(self, embed_shape,n_agents, state_shape):
        super(MixingNet, self).__init__()
        self.n_agents = n_agents
        self.embed_shape = embed_shape
        self.state_shape = state_shape

        self.hyper_w1 = Dense(self.n_agents*embed_shape,
                              input_shape=(self.state_shape,),
                              activation=None)
        self.hyper_w2 = Dense(embed_shape,input_shape=(self.state_shape,),activation=None)
        self.hyper_b1 = Dense(embed_shape,input_shape=(self.state_shape,),activation=None)
        self.hyper_b2 = ks.Sequential()
        self.hyper_b2.add(Dense(embed_shape,input_shape=(self.state_shape,),activation="relu"))
        self.hyper_b2.add(Dense(1,input_shape=(self.embed_shape,),activation=None))

    def call(self,inputs):
        q_values,state = inputs

        batch_size = q_values.shape[0]
        q_values = reshape(q_values, [batch_size,1,self.n_agents]) 

        w1 = tfabs(self.hyper_w1(state))   # (batch_size,n_agents*embed_shape)


        w1 = reshape(w1, [batch_size, self.n_agents, self.embed_shape]) # (batch_size,n_agents,embed_shape)
        b1 = self.hyper_b1(state)  # (batch_size,embed_shape)
        b1 = reshape(b1, [batch_size, 1, self.embed_shape])   # (batch_size,1,embed_shape)
        hidden = elu(matmul(q_values, w1) + b1)     # (batch_size,1,embed_shape)

        w2 = tfabs(self.hyper_w2(state))   # (batch_size,embed_shape)

        w2 = reshape(w2, [batch_size, self.embed_shape, 1])     # (batch_size,embed_shape,1)
        b2 = self.hyper_b2(state)      # (batch_size,1)
        b2 = reshape(b2, [batch_size, 1, 1])
        y = matmul(hidden, w2) + b2
        q_tot = reshape(y, [batch_size, 1])
        return q_tot



class QMIX:
    def __init__(self,
            observ_space: list,
            action_shape: list,
            args = None,
            act_only = False):


        self.name = "QMIX"
        self.model_dir = args.cwd

        self.recurrent = getattr(args,"if_recurrent",False)
        self.n_agents = getattr(args, "n_agents", 2)

        self.seq_len = getattr(args,"seq_len",3) if self.recurrent else None
        self.agents = [QAgent(action_shape[i],len(observ_space[i]),args,self.seq_len)
                    for i in range(self.n_agents)]
                    
        self.observ_space = observ_space
        self.action_shape = action_shape
        self.action_table = getattr(args,'action_table',None)
        self.state_shape = getattr(args,'state_shape')
        self.state_norm = array([[i for _ in range(self.state_shape)] for i in range(2)])
        self.if_norm = getattr(args,'if_norm',False)
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
            for agent in self.agents:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

            self.model = MixingNet(args.embed_shape,self.n_agents, args.state_shape)
            self.target_model = MixingNet(args.embed_shape,self.n_agents,args.state_shape)
            _ = self.model.call([tfrand.uniform([self.batch_size,self.n_agents]),
                            tfrand.uniform([self.batch_size,args.state_shape])])
            _ = self.target_model.call([tfrand.uniform([self.batch_size,self.n_agents]),
                            tfrand.uniform([self.batch_size,args.state_shape])])        
            
            self.trainable_variables += self.model.trainable_variables
            self.target_trainable_variables += self.target_model.trainable_variables

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
            if self.recurrent:
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
            loss = self._experience_replay(s, o, a, r, s_, o_, d)
            self.target_update_func()
            losses.append(loss)

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

        o,o_ = self._split_observ(s),self._split_observ(s_)
        loss = self._test_loss(s, o ,a ,r ,s_, o_ ,d)
        return loss


    def _normalize_state(self,s):
        # Normalize the state & reward
        s = ((array(s)-self.state_norm[0,:])/(self.state_norm[1,:]+1e-5)).tolist()
        # r = ((array(r)-self.reward_norm[0])/self.reward_norm[1]).tolist()
        return s

    def _split_observ(self,s):
        # Split as multi-agent & convert to tensor
        if self.recurrent:
            o = [convert_to_tensor([[[sis[idx] for idx in self.observ_space[i]]
                                   for sis in si] for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        else:
            o = [convert_to_tensor([[si[idx] for idx in self.observ_space[i]]
                                   for si in s],dtype=float32) 
                                   for i in range(self.n_agents)]
        return o


    def _experience_replay(self,s, o, a, r, s_, o_, d):
        o,o_ = [[convert_to_tensor(oi,dtype=float32) for oi in x] for x in [o,o_]]
        s,r,s_,d = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d]]
        a = convert_to_tensor(a)

        targets = self._calculate_target(r,s_,o_,d)

        with GradientTape() as tape:
            tape.watch(o)
            q_values = [reduce_sum(agent.forward(o[idx])*one_hot(a[:,idx],self.action_shape[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
            q_values = transpose(convert_to_tensor(q_values))
            q_tot = self.model.call([q_values,s]) 
            loss_value = self.loss_fn(targets, q_tot)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss_value.numpy()

    def _calculate_target(self,r,s_,o_,d):

        if self.double:
            target_q_values = [reduce_sum(agent.forward(o_[idx],target=True)*\
                one_hot(ks.backend.argmax(agent.forward(o_[idx])),self.action_shape[idx]),axis=1)
            for idx,agent in enumerate(self.agents)]
        else:
            target_q_values = [reduce_max(agent.forward(o_[idx],target=True),axis=1)
                    for idx,agent in enumerate(self.agents)]

        target_q_values = transpose(convert_to_tensor(target_q_values))
        target_q_tot = self.target_model.call([target_q_values,s_])

        discounted_reward_batch = self.gamma * target_q_tot
        targets = r + discounted_reward_batch * (1-d)
        return targets


    def _test_loss(self,s, o, a, r, s_, o_, d):
        o,o_ = [[convert_to_tensor(oi,dtype=float32) for oi in x] for x in [o,o_]]
        s,r,s_,d = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d]]
        a = convert_to_tensor(a)
        
        targets = self._calculate_target(r,s_,o_,d)
        
        q_values = [reduce_sum(agent.forward(o[idx])*one_hot(a[:,idx],self.action_shape[idx]),axis=1)
                    for idx,agent in enumerate(self.agents)]
        q_values = transpose(convert_to_tensor(q_values))
        q_tot = self.target_model.call([q_values,s])

        loss_value = self.loss_fn(targets, q_tot)

        return loss_value.numpy()
    

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
        model_dir = self.model_dir if model_dir is None else model_dir

        # Save the state normalization paras
        if norm:
            save(join(model_dir,'state_norm.npy'),self.state_norm)

        # Save the agent paras
        if agents:
            self.model.save_weights(join(model_dir,'mixing.h5'))
            self.target_model.save_weights(join(model_dir,'mixing_target.h5'))
            for i,agent in enumerate(self.agents):
                agent.save(i,model_dir)
            
    def load(self,model_dir=None,norm=True,agents=True):
        model_dir = self.model_dir if model_dir is None else model_dir
        # Load the state normalization paras
        if norm:
            self.state_norm = load(join(model_dir,'state_norm.npy'))

        # Load the agent paras
        if agents:
            for i,agent in enumerate(self.agents):
                agent.load(i,model_dir)

            if hasattr(self,'model'):
                self.model.build([TensorShape([self.batch_size,self.n_agents]),
                                TensorShape([self.batch_size,self.state_shape])])
                self.model.load_weights(join(model_dir,'mixing.h5'))

            if hasattr(self,'target_model'):
                self.target_model.build([TensorShape([self.batch_size,self.n_agents]),
                                        TensorShape([self.batch_size,self.state_shape])])  
                self.target_model.load_weights(join(model_dir,'mixing_target.h5'))


    