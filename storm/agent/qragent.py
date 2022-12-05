# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:02:01 2021

@author: MOMO
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,BatchNormalization,Input,Lambda,GRU
from tensorflow.keras.models import Model
from tensorflow import convert_to_tensor,expand_dims,concat
from numpy import array
import random
from os.path import join

class QRAgent:
    def __init__(self,action_shape,observ_size,seq_len,args):

        self.action_shape = action_shape
        self.observ_size = observ_size
        self.seq_len = seq_len

        self.net_dim = getattr(args,"net_dim",128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)
        self.dueling = getattr(args,"if_dueling",True)

        # deprecated
        self.epsilon = getattr(args, "epsilon", 1)
        self.epsilon_decay = getattr(args, "epsilon_decay", 0.999)
        self.epsilon_min = getattr(args, "epsilon_min", 0.1)

        self.update_interval = getattr(args, "update_interval", 0.005)

        self.target_model = self.build_q_network()
        self.model = self.build_q_network()
        self.target_model.set_weights(self.model.get_weights())
        self.model_dir = args.cwd
        # self.input_size = (seq_len,observ_size) if input_size is None else input_size
        # self.output_size = action_size if output_size is None else output_size

    def build_q_network(self):
        input_layer = Input(shape=(self.seq_len,self.observ_size,))
        x = Dense(self.net_dim, activation='relu')(input_layer)
        for _ in range(self.num_layer-1):
            x = Dense(self.net_dim, activation='relu')(x)
        x = GRU(self.hidden_dim)(x)
        if self.dueling:
            x = Dense(self.action_shape+1,activation = 'linear')(x)
            output = Lambda(lambda i: K.expand_dims(i[:,0],-1)+i[:,1:] - K.mean(i[:,1:],keepdims = True),
                            output_shape=(self.action_shape,))(x)
            # x = Dense(self.output_size+self.n_agents,activation = 'linear')(x)
            # output = Lambda(lambda i: concat([K.expand_dims(i[:,idx*(self.action_size+1)],-1)+\
            #                                              i[:,idx*(self.action_size+1)+1:(idx+1)*(self.action_size+1)] -\
            #                                                  K.mean(i[:,idx*(self.action_size+1)+1:(idx+1)*(self.action_size+1)],keepdims = True)
            #                                              for idx in self.n_agents],axis=-1),
            #                 output_shape=(self.output_size,))(x)            
        else:
            output = Dense(self.action_shape, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output)
        return model

        
        
    def act(self,observ):
        # x = convert_to_tensor(observ)
        # x = expand_dims(x,0)
        a = self.model(observ)[0].numpy().tolist()
        return a
    
    def _hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self,tau=None):
        tau = self.update_interval if tau is None else tau
        target_model_weights = array(self.target_model.get_weights())
        model_weights = array(self.model.get_weights())
        new_weight = (1. - tau) * target_model_weights \
            + tau * model_weights
        self.target_model.set_weights(new_weight)
    
    def save(self,i,model_dir=None):
        if model_dir is None:
            self.model.save_weights(join(self.model_dir,'agent%s.h5'%i))
            self.target_model.save_weights(join(self.model_dir,'agent%s_target.h5'%i))
        else:
            self.model.save_weights(join(model_dir,'agent%s.h5'%i))
            self.target_model.save_weights(join(model_dir,'agent%s_target.h5'%i))
            
    def load(self,i,model_dir=None):
        if model_dir is None:
            self.model.load_weights(join(self.model_dir,'agent%s.h5'%i))
            self.target_model.load_weights(join(self.model_dir,'agent%s_target.h5'%i))
        else:
            self.model.load_weights(join(model_dir,'agent%s.h5'%i))
            self.target_model.load_weights(join(model_dir,'agent%s_target.h5'%i))