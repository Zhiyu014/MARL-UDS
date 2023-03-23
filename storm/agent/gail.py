from tensorflow import keras as ks
from tensorflow import expand_dims,convert_to_tensor,float32,squeeze,GradientTape,reduce_mean,reduce_sum,one_hot
import tensorflow as tf
from numpy import array,save,load,zeros_like,concatenate
# import scipy.signal
from os.path import join

from utils.memory import RandomMemory
from .discirminator import Discriminator
from .ppo import PPO

class GAIL(PPO):
    def __init__(self,
            observ_space: int or list,
            action_shape: int or list,
            args = None,
            act_only = False):
        super().__init__(observ_space,action_shape,args,act_only)

        self.name = "GAIL"

        graph_conv = self.critic.conv_layer if self.graph_conv and self.share_conv_layer else self.graph_conv
        dis_out_shape = sum(self.action_shape) if self.if_mac else self.action_shape
        self.discri = Discriminator(dis_out_shape,self.state_shape,args,self.seq_len,graph_conv)


        if not act_only:
            self.dis_learning_rate = getattr(args,"dis_learning_rate",1e-3)
            self.dis_optimizer = ks.optimizers.get(args.optimizer)
            self.dis_optimizer.learning_rate = self.dis_learning_rate

            self.expert_traj_dir = args.expert_traj_dir
            self.load_experience()

        if args.if_load:
            self.load()
            
    def discri_forward(self,s,a):
        p = self.discri.forward(s)
        if self.if_mac:
            d = reduce_mean([reduce_sum(p[:,sum(self.action_shape[:i]):sum(self.action_shape[:i])+size]*one_hot(a[:,i],self.action_shape[i]),axis=1) for i,size in enumerate(self.action_shape)],axis=0)
        else:
            d = reduce_sum(p*one_hot(squeeze(a),self.action_shape),axis=1)
        return d


    def update_net(self,memory,batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        discri_losses,value_losses,policy_losses = [],[],[]
        for _ in range(self.repeat_times):
            s, a, _, s_, d, log_probs, value = memory.sample(batch_size,continuous=True)
            if self.if_mac:
                o = self._split_observ(s)
            s, s_,d, log_probs,value = [convert_to_tensor(i,dtype=float32) for i in [s,s_,d,log_probs,value]]
            a = convert_to_tensor(a)
            dis = self.discri_forward(s,a)
            r = tf.math.log(dis + 1e-5) - tf.math.log(1-dis + 1e-5)
            
            # Normalize the imitation reward
            # r = (r - reduce_mean(r))/(tf.math.reduce_std(r) + 1e-5)

            last_value = 0 if d[-1] is True else self.criticize(s_[-1])
            # deltas = r[:-1] + self.gamma * value[1:] * (1-d) - value[:-1]
            advs = self.get_advantages(r,d,value,last_value)
            # returns = self.discounted_cumsum(r,self.gamma)
            returns = advs + value

            discri_loss = self.discri_update(s,a)
            value_loss = self.critic_update(s, returns)
            policy_loss = [self.actor_update(o[i], a[:,i], log_probs[:,i], advs, i) for i in range(self.n_agents)] if self.if_mac else self.actor_update(s, a, log_probs, advs)

            self.target_update_func()
            discri_losses.append(discri_loss)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)
        return policy_losses


    def discri_update(self,s_f,a_f):
        s_t,a_t,_,_,_ = self.expert_trajs.sample(s_f.shape[0],continuous=True)
        s_t,a_t = convert_to_tensor(s_t,dtype=float32),convert_to_tensor(a_t)
        with GradientTape() as tape:
            tape.watch(self.discri.model.trainable_variables)
            d_t,d_f = self.discri_forward(s_t,a_t),self.discri_forward(s_f,a_f)
            # discri_loss = reduce_mean(tf.math.log(1-d_t)) + reduce_mean(tf.math.log(d_f))
            discri_loss = ks.losses.binary_crossentropy(d_t, tf.ones_like(d_t))
            discri_loss += ks.losses.binary_crossentropy(d_f, tf.zeros_like(d_f))
        grads = tape.gradient(discri_loss, self.discri.model.trainable_variables)
        self.dis_optimizer.apply_gradients(zip(grads, self.discri.model.trainable_variables))
        return discri_loss

    def evaluate_net(self,trajs):
        s, a = [[traj[i] for traj in trajs] for i in range(2)]
        s,a = convert_to_tensor(s,dtype=float32),convert_to_tensor(a)
        dis = self.discri_forward(s,a)
        r = (tf.math.log(dis) - tf.math.log(1-dis)).numpy().tolist()
        for i,ri in enumerate(r):
            trajs[i][2] = ri
        return super().evaluate_net(trajs)


    def save(self,model_dir=None,norm=False,agents=True):
        # Save the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                save(join(self.model_dir,'state_norm.npy'),self.state_norm)
            else:
                save(join(model_dir,'state_norm.npy'),self.state_norm)
        # Load the agent paras
        if agents:
            if self.if_mac:
                for i,actor in enumerate(self.actor):
                    actor.save(i,model_dir)
            else:
                self.actor.save(0,model_dir)
            self.critic.save(0,model_dir)
            self.discri.save(0,model_dir)


    def load_experience(self,cwd=None):
        expert_traj_dir = self.expert_traj_dir if cwd is None else cwd
        self.expert_trajs = RandomMemory(2**20,cwd=expert_traj_dir,load=True)

    def load(self,model_dir=None,norm=False,agents=True):
        # Load the state normalization paras
        if norm and self.if_norm:
            if model_dir is None:
                self.state_norm = load(join(self.model_dir,'state_norm.npy'))
            else:
                self.state_norm = load(join(model_dir,'state_norm.npy'))
        # Load the agent paras
        if agents:
            if self.if_mac:
                for i,actor in enumerate(self.actor):
                    actor.load(i,model_dir)
            else:
                self.actor.load(0,model_dir)
            self.critic.load(0,model_dir)
            if hasattr(self,'discri'):
                self.discri.load(0,model_dir)
