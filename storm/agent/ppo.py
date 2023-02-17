from tensorflow import keras as ks
from tensorflow import expand_dims,convert_to_tensor,float32,squeeze,GradientTape,reduce_mean
import tensorflow as tf
from numpy import array,save,load,zeros_like,concatenate
# import scipy.signal
from os.path import join
from .piagent import Actor
from .qagent import QAgent

class PPO:
    def __init__(self,
            observ_space: int or list,
            action_shape: int or list,
            args = None,
            act_only = False):

        self.name = "PPO"
        self.model_dir = args.cwd

        self.recurrent = getattr(args,"if_recurrent",False)
        self.n_agents = getattr(args, "n_agents", 1)
        self.state_shape = getattr(args, "state_shape", 10)
        self.observ_space = observ_space
        self.action_shape = action_shape
        self.if_mac = getattr(args,'if_mac',False)            
        self.if_norm = getattr(args,'if_norm',False)

        self.seq_len = getattr(args,"seq_len",3) if self.recurrent else None
        self.graph_conv = getattr(args,"global_state",False)
        self.share_conv_layer = getattr(args,"share_conv_layer",False)
        self.critic = QAgent(1,self.state_shape,args,self.seq_len,self.graph_conv) 
        if self.if_mac:
            self.actor = [Actor(self.action_shape[i],len(self.observ_space[i]),args,self.seq_len) 
            for i in range(self.n_agents)]
        else:
            graph_conv = self.critic.conv_layer if self.graph_conv and self.share_conv_layer else self.graph_conv
            self.actor = Actor(action_shape,observ_space,args,self.seq_len,graph_conv)

        self.action_table = getattr(args,'action_table',None)

        if not act_only:
            self.horizon_len = getattr(args,"horizon_len",6)
            self.gamma = getattr(args, "gamma", 0.98)
            self.lambda_gae = getattr(args, "lambda_gae", 0.95)
            self.lambda_entropy = getattr(args, "lambda_entropy", 0)
            self.clip_ratio = getattr(args, "clip_ratio", 0.2)
            self.batch_size = getattr(args,"batch_size",256)
            self.act_learning_rate = getattr(args,"act_learning_rate",1e-4)
            self.cri_learning_rate = getattr(args,"cri_learning_rate",1e-3)
            self.repeat_times = getattr(args,"repeat_times",2)
            self.update_interval = getattr(args,"update_interval",0.005)
            self.target_update_func = self._hard_update_target_model if self.update_interval >1\
                else self._soft_update_target_model
            self.episode = getattr(args,'episode',0)

            self.loss_fn = ks.losses.get(args.loss_function)
            self.act_optimizer = ks.optimizers.get(args.optimizer)
            self.cri_optimizer = ks.optimizers.get(args.optimizer)
            self.act_optimizer.learning_rate = self.act_learning_rate
            self.cri_optimizer.learning_rate = self.cri_learning_rate

        if args.if_load:
            self.load()


    def act(self,state,train=True):
        if self.recurrent:
            state =  [state[0] for _ in range(self.seq_len-len(state))]+state \
                if len(state)<self.seq_len else state
        # Get action and logp
        if self.if_mac:
            observ = self._split_observ([state])
            res = [act.get_action(observ[i],train) for i,act in enumerate(self.actor)]
            a = concatenate([r[0].numpy() for r in res],axis=0)
            log_probs = concatenate([r[1].numpy() for r in res],axis=0)
        else:
            state = expand_dims(convert_to_tensor(state),0)
            a,log_probs = self.actor.get_action(state,train)
            a,log_probs = a.numpy(),log_probs.numpy()
        # a,probs = self.actor.get_action_value(state,train)

        # if self.if_mac:
        #     action = []
        #     for i,shape in enumerate(self.action_shape):
        #         j = sum(self.action_shape[:i])
        #         action.append(a[j:j + shape])
        return a,log_probs

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

    def criticize(self,state):
        if self.recurrent:
            state =  [state[0] for _ in range(self.seq_len-len(state))]+state \
                if len(state)<self.seq_len else state
        # Get action and logp
        state = expand_dims(convert_to_tensor(state),0)
        value = squeeze(self.critic.forward(state))
        return value
        
    def convert_action_to_setting(self,action):
        if self.action_table is not None:
            setting = self.action_table[tuple(action)]
            return setting
        else:
            setting = [int(act) for act in action]
            return setting

    def get_advantages(self,r,d,value,last_value):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        # from ElegantRL
        # TODO: debug
        advs = zeros_like(r)
        bs = r.shape[0]
        iters = bs // self.horizon_len + 1
        for i in range(iters):
            idx0,idx1 = i*self.horizon_len,min(bs,(i+1)*self.horizon_len)
            next_ = value[idx1] if idx1 < bs else last_value
            adv = 0
            for t in range(min(bs,idx1)-1, idx0-1, -1):
                next_ = r[t] + self.gamma*(1-d[t]) * next_
                advs[t] = adv = next_ - value[t] + self.gamma*self.lambda_gae*(1-d[t])*adv
                next_ = value[t]
        return convert_to_tensor(advs)
            
    def update_net(self,memory,batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        value_losses,policy_losses = [],[]
        for _ in range(self.repeat_times):
            s, a, r, s_, d, log_probs, value = memory.sample(batch_size,continuous=True)
            if self.if_mac:
                o = self._split_observ(s)
            s,r, s_,d, log_probs,value = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d,log_probs,value]]
            a = convert_to_tensor(a)

            last_value = 0 if d[-1] is True else self.criticize(s_[-1])
            # deltas = r[:-1] + self.gamma * value[1:] * (1-d) - value[:-1]
            advs = self.get_advantages(r,d,value,last_value)
            # returns = self.discounted_cumsum(r,self.gamma)
            returns = advs + value

            value_loss = self.critic_update(s, returns)
            policy_loss = [self.actor_update(o[i], a[:,i], log_probs[:,i], advs, i) for i in range(self.n_agents)] if self.if_mac else self.actor_update(s, a, log_probs, advs)

            self.target_update_func()
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)
        return policy_losses

    def evaluate_net(self,trajs):
        s, a, r, s_, d, log_probs, value = [[traj[i] for traj in trajs] for i in range(7)]
        if self.recurrent:
            s = [[s[0] for _ in range(self.seq_len-i-1)]+s[:i+1] for i in range(self.seq_len-1)]+\
                [s[i:i+self.seq_len] for i in range(len(s)-self.seq_len+1)]
            s_ = [[s_[0] for _ in range(self.seq_len-i-1)]+s_[:i+1] for i in range(self.seq_len-1)]+\
                [s_[i:i+self.seq_len] for i in range(len(s_)-self.seq_len+1)]      
        if self.if_mac:
            o = self._split_observ(s)
        s,r, s_,d, log_probs,value = [convert_to_tensor(i,dtype=float32) for i in [s,r,s_,d,log_probs,value]]
        a = convert_to_tensor(a)

        last_value = 0 if d[-1] is True else self.criticize(s_[-1])
        # deltas = r[:-1] + self.gamma * value[1:] * (1-d) - value[:-1]
        advs = self.get_advantages(r,d,value,last_value)
        # returns = self.discounted_cumsum(r,self.gamma)
        returns = advs + value

        y_preds = squeeze(self.critic.forward(s))
        value_loss = self.loss_fn(y_preds, returns)

        min_adv = tf.where(
            advs > 0,
            (1 + self.clip_ratio) * advs,
            (1 - self.clip_ratio) * advs
        )
        if self.if_mac:
            policy_loss = []
            for i,actor in enumerate(self.actor):
                new_log_probs,entropy = actor.get_action_entropy(o[i],a[:,i])
                ratio = tf.exp(new_log_probs - log_probs[:,i])
                loss = - reduce_mean(tf.minimum(ratio * advs, min_adv))
                loss -= reduce_mean(entropy) * self.lambda_entropy # entropy loss
                policy_loss.append(loss)
        else:
            new_log_probs,entropy = self.actor.get_action_entropy(s,a)
            ratio = tf.exp(new_log_probs - log_probs)
            policy_loss = - reduce_mean(tf.minimum(ratio * advs, min_adv))
            policy_loss -= reduce_mean(entropy) * self.lambda_entropy # entropy loss
        return policy_loss


    def critic_update(self,s, returns):
        with GradientTape() as tape:
            tape.watch(s)
            y_preds = squeeze(self.critic.forward(s))
            value_loss = self.loss_fn(y_preds, returns)
        grads = tape.gradient(value_loss, self.critic.model.trainable_variables)
        self.cri_optimizer.apply_gradients(zip(grads, self.critic.model.trainable_variables))
        return value_loss.numpy()

    def actor_update(self,s, a, log_probs, advs,i=None):
        actor = self.actor[i] if self.if_mac else self.actor
        variables = actor.model.trainable_variables
        with GradientTape() as tape:
            tape.watch(s)
            new_log_probs,entropy = actor.get_action_entropy(s,a)
            ratio = tf.exp(new_log_probs - log_probs)
            min_adv = tf.where(
                advs > 0,
                (1 + self.clip_ratio) * advs,
                (1 - self.clip_ratio) * advs
            )
            policy_loss = - reduce_mean(tf.minimum(ratio * advs, min_adv))
            policy_loss -= reduce_mean(entropy) * self.lambda_entropy # entropy loss
        grads = tape.gradient(policy_loss, variables)
        self.act_optimizer.apply_gradients(zip(grads, variables))
        return policy_loss.numpy()

    def episode_update(self,episode):
        self.episode = episode

    def _hard_update_target_model(self):
        if self.episode%self.update_interval == 0:
            self.critic._hard_update_target_model()

    def _soft_update_target_model(self):
        self.critic._soft_update_target_model()


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