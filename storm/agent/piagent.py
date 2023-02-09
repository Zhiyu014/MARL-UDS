from tensorflow.keras.layers import Dense,Input,GRU
from tensorflow.keras.models import Model
from spektral.layers import GCNConv,GlobalAttnSumPool
from spektral.utils.convolution import gcn_filter
from numpy import zeros
from os.path import join
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

class Actor:
    def __init__(self,action_shape,observ_size,args,seq_len=None,graph_conv=False):
        self.action_shape = action_shape
        self.observ_size = observ_size
        self.recurrent = True if seq_len != None else False
        self.seq_len = seq_len

        self.net_dim = getattr(args,"net_dim",128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)

        # Use Graph convolution
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

        self.model = self.build_pi_network(self.conv_layer)
        self.model_dir = args.cwd

    def build_pi_network(self,conv=None):
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
        output = Dense(self.action_shape, activation='softmax')(x)
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

    # def build_pi_network(self):
    #     input_shape = (self.observ_size,) if isinstance(self.observ_size,int) else tuple(self.observ_size)
    #     if self.recurrent:
    #         input_shape = (self.seq_len,) + input_shape
    #     input_observ = Input(shape=input_shape)
    #     if self.graph_conv:
    #         input_A = Input(shape=(self.graph_filter.shape[0],))
    #         x = GCNConv(self.graph_channel,activation='relu')([input_observ,input_A])
    #         for _ in range(self.num_conv_layer-1):
    #             x = GCNConv(self.graph_channel,activation='relu')([x,input_A])
    #         x = GlobalAttnSumPool()(x)
    #     else:
    #         x = Dense(self.net_dim, activation='relu')(input_observ)
    #     for _ in range(self.num_layer-1):
    #         x = Dense(self.net_dim, activation='relu')(x)
    #     if self.recurrent:
    #         x = GRU(self.hidden_dim)(x)
    #     output = Dense(self.action_shape, activation='softmax')(x)
    #     input_layer = [input_observ,input_A] if self.graph_conv else input_observ
    #     model = Model(inputs=input_layer, outputs=output)
    #     return model

    def act(self, observ):
        inp = [observ,self.graph_filter] if self.graph_conv else observ
        pi = self.model(inp)[0].numpy().tolist()
        return pi

    def forward(self,observ):
        inp = [observ,self.graph_filter] if self.graph_conv else observ
        probs = self.model(inp)
        return probs


    def get_action(self, observ, train = True):
        probs = self.forward(observ)
        m = Categorical(probs)
        if train:
            action = m.sample()
        else:
            action = tf.argmax(m.logits,axis=1)
        logp_action = m.log_prob(action)
        return action,logp_action

    def get_action_entropy(self,observ,action):
        probs = self.forward(observ)
        m = Categorical(probs)
        logp_action = m.log_prob(action)
        entrophy = m.entropy()
        return logp_action,entrophy

    def get_action_value(self, observ, train=True):
        probs = self.forward(observ)
        if train:
            action = tf.squeeze(tf.random.categorical(probs,1))
        else:
            action = tf.squeeze(tf.argmax(probs,axis=1))
        return action,probs

    def save(self,i,model_dir=None):
        if model_dir is None:
            self.model.save_weights(join(self.model_dir,'actor%s.h5'%i))
        else:
            self.model.save_weights(join(model_dir,'actor%s.h5'%i))
            
    def load(self,i,model_dir=None):
        if model_dir is None:
            self.model.load_weights(join(self.model_dir,'actor%s.h5'%i))
        else:
            self.model.load_weights(join(model_dir,'actor%s.h5'%i))


    # def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
    #     state = self.state_norm(state)
    #     action_avg = self.net(state)
    #     action_std = self.action_std_log.exp()

    #     dist = self.ActionDist(action_avg, action_std)
    #     logprob = dist.log_prob(action).sum(1)
    #     entropy = dist.entropy().sum(1)
    #     return logprob, entropy