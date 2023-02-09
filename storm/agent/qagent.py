# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:02:01 2021

@author: MOMO
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Input,Lambda,GRU
from tensorflow.keras.models import Model
from spektral.layers import GCNConv,GlobalAttnSumPool
from spektral.utils.convolution import gcn_filter
from numpy import array,zeros
from os.path import join

class QAgent:
    def __init__(self,action_shape,observ_size,args,seq_len=None,graph_conv=False):
        self.action_shape = action_shape
        self.observ_size = observ_size
        self.recurrent = True if seq_len != None else False
        self.seq_len = seq_len

        self.net_dim = getattr(args,"net_dim",128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)
        self.dueling = getattr(args,"if_dueling",False)

        # TODO:Use Graph convolution; Shared conv layer
        if graph_conv:
            edges = getattr(args,'edges')
            A = zeros((edges.max()+1,edges.max()+1)) # adjacency matrix
            for u,v in edges:
                A[u,v] += 1
            self.graph_filter = gcn_filter(A)
            if graph_conv == True:
                self.graph_channel = getattr(args,"graph_channel",self.net_dim)
                self.num_conv_layer = getattr(args, "num_conv_layer", self.num_layer)
                self.conv_layer = self.build_conv_model()
            else:
                self.conv_layer = graph_conv
        else:
            self.conv_layer = None
        self.graph_conv = bool(graph_conv)

        self.update_interval = getattr(args, "update_interval", 0.005)
        self.target_model = self.build_q_network(self.conv_layer)
        self.model = self.build_q_network(self.conv_layer)
        self.target_model.set_weights(self.model.get_weights())
        self.model_dir = args.cwd
        
    def build_q_network(self,conv=None):
        if conv is None:
            input_shape = (self.seq_len,self.observ_size) if self.recurrent else (self.observ_size,)
            x_in = Input(shape=input_shape)
            x = x_in
        else:
            x_in,x = conv.input,conv.output
        for _ in range(self.num_layer):
            x = Dense(self.net_dim, activation='relu')(x)
        if self.recurrent:
            x = GRU(self.hidden_dim)(x)
        if self.dueling:
            x = Dense(self.action_shape+1,activation = 'linear')(x)
            output = Lambda(lambda i: K.expand_dims(i[:,0],-1)+i[:,1:] - K.mean(i[:,1:],keepdims = True),
                            output_shape=(self.action_shape,))(x)
        else:
            output = Dense(self.action_shape, activation='linear')(x)
        model = Model(inputs=x_in, outputs=output)
        return model

    def build_conv_model(self):
        input_shape = tuple(self.observ_size)
        if self.recurrent:
            input_shape = (self.seq_len,) + input_shape
        input_observ = Input(shape=input_shape)
        input_A = Input(shape=(self.graph_filter.shape[0],))
        x = GCNConv(self.graph_channel,activation='relu')([input_observ,input_A])
        for _ in range(self.num_conv_layer-1):
            x = GCNConv(self.graph_channel,activation='relu')([x,input_A])
        x = GlobalAttnSumPool()(x)
        return Model([input_observ,input_A],x)


    def forward(self,observ,target=False):
        inp = [observ,self.graph_filter] if self.graph_conv else observ
        q = self.target_model(inp) if target else self.model(inp)
        return q
        
    def act(self,observ):
        inp = [observ,self.graph_filter] if self.graph_conv else observ
        q = self.model(inp)[0].numpy().tolist()
        return q
    
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