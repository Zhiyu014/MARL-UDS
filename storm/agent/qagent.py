# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:02:01 2021

@author: MOMO
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,BatchNormalization,Input,Lambda,GRU
from tensorflow.keras.models import Model
from tensorflow import convert_to_tensor,expand_dims
from numpy import array
import random
from os.path import join

class QAgent:
    def __init__(self,action_shape,observ_size,args):
        self.action_shape = action_shape
        self.observ_size = observ_size

        self.net_dim = getattr(args,"net_dim",128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.dueling = getattr(args,"if_dueling",True)

        self.epsilon = getattr(args, "epsilon", 1)
        self.epsilon_decay = getattr(args, "epsilon_decay", 0.999)
        self.epsilon_min = getattr(args, "epsilon_min", 0.1)

        self.update_interval = getattr(args, "update_interval", 0.005)

        self.target_model = self.build_q_network()
        self.model = self.build_q_network()
        self.target_model.set_weights(self.model.get_weights())
        self.model_dir = args.cwd
        
    def build_q_network(self):
        input_layer = Input(shape=(self.observ_size,))
        x = Dense(self.net_dim, activation='relu')(input_layer)
        for _ in range(self.num_layer-1):
            x = Dense(self.net_dim, activation='relu')(x)
        if self.dueling:
            x = Dense(self.action_shape+1,activation = 'linear')(x)
            output = Lambda(lambda i: K.expand_dims(i[:,0],-1)+i[:,1:] - K.mean(i[:,1:],keepdims = True),
                            output_shape=(self.action_shape,))(x)
        else:
            output = Dense(self.action_shape, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output)
        return model

        
        
    def act(self,observ,train=True):
        x = convert_to_tensor(observ)
        x = expand_dims(x,0)
        if train:
            if random.random() > self.epsilon:
                # Get action from Q table
                a = self.model(x)[0].numpy().tolist()
            else:
                # Get random action
                a = [random.random() for _ in range(self.action_shape)]
        else:
            a = self.model(x)[0].numpy().tolist()
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
    
    def _epsilon_update(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay,self.epsilon_min)
        
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