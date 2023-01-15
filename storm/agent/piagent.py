from tensorflow.keras.layers import Dense,Input,GRU
from tensorflow.keras.models import Model
from os.path import join
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

class Actor:
    def __init__(self,action_shape,observ_size,args,seq_len=None):
        self.action_shape = action_shape
        self.observ_size = observ_size
        self.recurrent = True if seq_len != None else False
        self.seq_len = seq_len

        self.net_dim = getattr(args,"net_dim",128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.hidden_dim = getattr(args,"hidden_dim",self.net_dim)

        self.model = self.build_pi_network()
        self.model_dir = args.cwd

    def build_pi_network(self):
        if self.recurrent:
            input_layer = Input(shape=(self.seq_len,self.observ_size,))
        else:
            input_layer = Input(shape=(self.observ_size,))
        x = Dense(self.net_dim, activation='relu')(input_layer)
        for _ in range(self.num_layer-1):
            x = Dense(self.net_dim, activation='relu')(x)
        if self.recurrent:
            x = GRU(self.hidden_dim)(x)
        output = Dense(self.action_shape, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        return model

    def act(self, observ):
        pi = self.model(observ)[0].numpy().tolist()
        return pi

    def get_action(self, observ, train = True):
        probs = self.model(observ)
        m = Categorical(probs)
        if train:
            action = m.sample()
        else:
            action = tf.argmax(m.logits,axis=1)
        logp_action = m.log_prob(action)
        return action,logp_action

    def get_action_entropy(self,observ,action):
        probs = self.model(observ)
        m = Categorical(probs)
        logp_action = m.log_prob(action)
        entrophy = m.entropy()
        return logp_action,entrophy

    def get_action_value(self, observ, train=True):
        probs = self.model(observ)
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