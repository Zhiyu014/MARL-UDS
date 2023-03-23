from tensorflow.keras.layers import Dense,Input,GRU
from tensorflow.keras.models import Model,Sequential
from tensorflow import keras as ks
from tensorflow import one_hot,expand_dims,convert_to_tensor,float32,squeeze,GradientTape,reduce_mean
from numpy import array,save,load,zeros_like,concatenate
from os.path import join
from .piagent import Actor
from utils.memory import RandomMemory

class Behavior_cloning:
    def __init__(self,
                 observ_space,
                 action_shape,
                 args = None,
                 act_only = False):
        self.name = "Behavior_cloning"
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
        if self.if_mac:
            self.actor = [Actor(self.action_shape[i],len(self.observ_space[i]),args,self.seq_len) 
            for i in range(self.n_agents)]
        else:
            self.actor = Actor(action_shape,observ_space,args,self.seq_len,self.graph_conv)

        self.action_table = getattr(args,'action_table',None)


        if not act_only:
            self.batch_size = getattr(args,"batch_size",256)
            self.epochs = getattr(args,'epochs',100)
            self.learning_rate = getattr(args,"learning_rate",1e-3)
            self.metric = ks.metrics.get(args.metric)
            self.loss_fn = ks.losses.get(args.loss_function)
            self.optimizer = ks.optimizers.get(args.optimizer)
            self.optimizer.learning_rate = self.learning_rate

            self.expert_traj_dir = args.expert_traj_dir
            self.load_experience()

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
    
    def convert_action_to_setting(self,action):
        if self.action_table is not None:
            setting = self.action_table[tuple(action)]
            return setting
        else:
            setting = [int(act) for act in action]
            return setting
        
    def update_net(self,epochs = None,batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        epochs = self.epochs if epochs is None else epochs
        for epoch in range(epochs):
            s,a = self.expert_trajs.sample(batch_size)[:2]
            s,a = convert_to_tensor(s,dtype=float32),convert_to_tensor(a)
            if self.if_mac:
                o = self._split_observ(s)
                losses = []
                accs = []
                for i,act in enumerate(self.actor):
                    with GradientTape() as tape:
                        tape.watch(act.model.trainable_variables)
                        ai = one_hot(a[:,i],self.action_shape[i])
                        a_pred = act.model(o[i])
                        loss = self.loss_fn(a_pred,ai)
                    grads = tape.gradient(loss, act.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, act.model.trainable_variables))
                    losses.append(loss.numpy())
                    accs.append(self.metric(a_pred,ai))
                loss,acc = reduce_mean(losses), reduce_mean(accs)
            else:
                with GradientTape() as tape:
                    tape.watch(self.actor.model.trainable_variables)
                    ai = one_hot(a,self.action_shape)
                    a_pred = self.actor.model(s)
                    loss = self.loss_fn(a_pred,ai)
                grads = tape.gradient(loss, self.actor.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))
                acc = self.metric(a_pred,ai)
            print("Epoch {}/{} loss: {} Metric: {}".format(epoch,epochs,loss.numpy(),acc.numpy()))
        return loss.numpy()
    
    def evaluate_net(self,trajs):
        s,a = trajs[:2]
        s,a = convert_to_tensor(s,dtype=float32),convert_to_tensor(a)
        if self.if_mac:
            o = self._split_observ(s)
            losses = []
            accs = []
            for i,act in enumerate(self.actor):
                ai = one_hot(a[:,i],self.action_shape[i])
                a_pred = act.model(o[i])
                loss = self.loss_fn(a_pred,ai)
                losses.append(loss.numpy())
                accs.append(self.metric(a_pred,ai))
            loss,acc = reduce_mean(losses), reduce_mean(accs)
        else:
            ai = one_hot(a,self.action_shape)
            a_pred = self.actor.model(s)
            loss = self.loss_fn(a_pred,ai)
            acc = self.metric(a_pred,ai)
        print("Testing loss: {} metric: {}".format(loss.numpy(),acc.numpy()))
        return loss.numpy()
    

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
