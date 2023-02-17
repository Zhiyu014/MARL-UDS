from tensorflow.keras.layers import Dense,Input,GRU
from tensorflow.keras.models import Model
from spektral.layers import GCNConv,GlobalAttnSumPool
from spektral.utils.convolution import gcn_filter
from numpy import zeros
from os.path import join
from tensorflow import one_hot,reduce_sum

class Discriminator:
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

        self.model = self.build_dis_network(self.conv_layer)
        self.model_dir = args.cwd

    def build_dis_network(self,conv = None):
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
        output = Dense(self.action_shape, activation='sigmoid')(x)
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

    def forward(self,observ):
        inp = [observ,self.graph_filter] if self.graph_conv else observ
        probs = self.model(inp)
        return probs

    def save(self,i,model_dir=None):
        if model_dir is None:
            self.model.save_weights(join(self.model_dir,'discriminator%s.h5'%i))
        else:
            self.model.save_weights(join(model_dir,'discriminator%s.h5'%i))
            
    def load(self,i,model_dir=None):
        if model_dir is None:
            self.model.load_weights(join(self.model_dir,'discriminator%s.h5'%i))
        else:
            self.model.load_weights(join(model_dir,'discriminator%s.h5'%i))