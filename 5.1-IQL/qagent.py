# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:02:01 2021

@author: MOMO
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,BatchNormalization,Input,Lambda
from tensorflow.keras.models import Model
from tensorflow import convert_to_tensor,expand_dims
from numpy import array
import random

class QAgent:
    def __init__(self,epsilon_decay,action_size,observ_size,dueling = False):
        self.action_size = action_size
        self.observ_size = observ_size
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.target_model = self.build_q_network(dueling)
        self.model = self.build_q_network(dueling)
        self.target_model.set_weights(self.model.get_weights())
        # self.trajectory = deque(maxlen=int(timesteps))
        
    def build_q_network(self,dueling):
        input_layer = Input(shape=(self.observ_size,))
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        if dueling:
            x = Dense(self.action_size+1,activation = 'linear')(x)
            output = Lambda(lambda i: K.expand_dims(i[:,0],-1)+i[:,1:] - K.mean(i[:,1:],keepdims = True), output_shape=(self.action_size,))(x)
        else:
            output = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    def observe(self,observ):
        self.state = observ
        self.trajectory.append(observ)
        
        
    def act(self,observ,train=True):
        x = convert_to_tensor(observ)
        x = expand_dims(x,0)
        if train:
            if random.random() > self.epsilon:
                # Get action from Q table
                a = self.model(x)[0].numpy().tolist()
            else:
                # Get random action
                a = [random.random() for _ in range(self.action_size)]
        else:
            a = self.model(x)[0].numpy().tolist()
        return a
    
    def _hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = array(self.target_model.get_weights())
        model_weights = array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)
    
    def _epsilon_update(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay,self.epsilon_min)
        
    def save(self,i):
        self.model.save_weights('./model/agent%s.h5'%i)
        self.target_model.save_weights('./model/agent%s_target.h5'%i)

    def load(self,i):
        self.model.load_weights('./model/agent%s.h5'%i)
        self.target_model.load_weights('./model/agent%s_target.h5'%i)