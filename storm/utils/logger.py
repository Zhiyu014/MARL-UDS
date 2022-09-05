
import numpy as np
from functools import reduce
import os
import json
from matplotlib import pyplot as plt
plt.rc('font',family = 'Times New Roman')

class Trainlogger:
    def __init__(self,model_dir,load=False):
        self.attrs = ['{0}_{1}'.format(k,v)
                         for k in ['train','eval'] 
                         for v in ['perfs','loss']]
        self.attrs += ['train_rewards']
        for attr in self.attrs:
            setattr(self,attr,list())

        self.max_train_rewards = 0
        self.min_train_perf = 1e10
        self.min_eval_perf = 1e10

        self.cwd = model_dir
        if load:
            self.load()

    def log(self,data,train=False):
        if train:
            self.train_rewards.append(data[0])
            self.train_perfs.append(data[1])
            self.train_loss.append(data[2])
            update = (sum(data[0]) > self.max_train_rewards, 
            sum(data[1]) < self.min_train_perf)
            self.max_train_rewards = max(self.max_train_rewards,sum(data[0]))
            self.min_train_perf = min(self.min_train_perf,sum(data[1]))
        else:
            self.eval_perfs.append(data[0])
            self.eval_loss.append(data[1])
            update = sum(data[0]) < self.min_eval_perf
            self.min_eval_perf = min(self.min_eval_perf,sum(data[0]))
        return update

    def save(self,model_dir=None):
        model_dir = self.cwd if model_dir is None else model_dir
        for attr in self.attrs:
            value = getattr(self,attr)
            np.save(os.path.join(model_dir,'%s.npy'%attr),value)

    def load(self,model_dir=None):
        model_dir = self.cwd if model_dir is None else model_dir
        for attr in self.attrs:
            value = np.load(os.path.join(model_dir,'%s.npy'%attr),allow_pickle=True).tolist()
            setattr(self,attr,value)
        self.min_train_perf = min([sum(data) for data in self.train_perfs])
        self.min_eval_perf = min([sum(data) for data in self.eval_perfs])
        self.max_train_rewards = max([sum(data) for data in self.train_rewards])


    def plot(self,filedir=None,event_wise=False,agent_wise=False):
        fig,((axL,axP),(axM,axR)) = plt.subplots(nrows=2,ncols=2,figsize = (10,10), dpi=600)

        if event_wise:
            for idx in range(len(self.train_perfs[0])):
                perf = [per[idx] for per in self.train_perfs]
                axL.plot(np.arange(len(perf)),perf,label = 'event_%s'%idx)
        else:
            train_perfs = np.array(self.train_perfs).sum(axis=1)
            axL.plot(np.arange(len(train_perfs)),train_perfs,label = 'perf')
        axL.set_xlabel('episode')
        axL.set_title('episode performance')
        axL.legend(loc='upper right')

        # for idx in range(vdnn.n_agents):
        #     axP.plot(arange(len(train_loss_history)),[loss[idx] for loss in train_loss_history],label='Agent %s'%idx)
        train_loss = reduce(lambda x,y:x+y, self.train_loss)
        if len(np.array(train_loss).shape) > 1:
            if agent_wise:
                for idx in range(len(train_loss[0])):
                    loss = [los[idx] for los in train_loss]
                    axP.plot(np.arange(len(loss)),loss,label='agent%s'%idx)
            else:
                train_loss = np.array(train_loss).mean(axis=1)
                axP.plot(np.arange(len(train_loss)),train_loss,label='average loss')
        else:
            axP.plot(np.arange(len(train_loss)),train_loss,label='average loss')
        axP.set_xlabel('Update times')
        axP.set_title("training loss")
        axP.legend(loc='upper right')


        if event_wise:
            for idx in range(len(self.eval_perfs[0])):
                perf = [per[idx] for per in self.eval_perfs]
                axM.plot(np.arange(len(perf)),perf,label = 'event_%s'%idx)
        else:
            eval_perfs = np.array(self.eval_perfs).sum(axis=1)
            axM.plot(np.arange(len(eval_perfs)),eval_perfs,label = 'perf')
        axM.set_xlabel('episode')
        axM.set_title('eval performance')
        axM.legend(loc='upper right')


        if event_wise:
            for idx in range(len(self.eval_loss[0])):
                loss = [np.array(los[idx]).mean() for los in self.eval_loss]
                axR.plot(np.arange(len(loss)),loss,label = 'event_%s'%idx)
        elif agent_wise:
            eval_loss = np.array(self.eval_loss).mean(axis=1)
            for idx in range(eval_loss.shape[-1]):
                loss = eval_loss[:,idx]
                axR.plot(np.arange(len(loss)),loss,label = 'agent_%s'%idx)
        else:
            eval_loss = [np.array(loss).mean() for loss in self.eval_loss]
            axR.plot(np.arange(len(eval_loss)),eval_loss,label = 'loss')
        axR.set_xlabel('episode')
        axR.set_title("eval loss")
        axR.legend(loc='upper right')


        fig2,ax = plt.subplots(nrows=1,ncols=1,figsize = (5,5), dpi=600)
        if event_wise:
            for idx in range(len(self.train_rewards[0])):
                rewards = [per[idx] for per in self.train_rewards]
                ax.plot(np.arange(len(rewards)),rewards,label = 'event_%s'%idx)
        else:
            train_rewards = np.array(self.train_rewards).sum(axis=1)
            ax.plot(np.arange(len(train_rewards)),train_rewards,label = 'rewards')
        ax.set_xlabel('episode')
        ax.set_title('train rewards')
        ax.legend(loc='lower right')


        if filedir is None:
            fig.savefig(os.path.join(self.cwd,'training.png'))
            fig2.savefig(os.path.join(self.cwd,'rewards.png'))

        elif os.path.isdir(filedir):
            fig.savefig(os.path.join(filedir,'training.png'))
            fig2.savefig(os.path.join(filedir,'rewards.png'))

        else:
            fig.savefig(filedir)
            fig2.savefig(filedir)


class Testlogger:
    def __init__(self,model_dir,load=False):
        self.records = {}
        self.cwd = model_dir
        self.event = None
        if load:
            self.load()

    def log(self,data,name,event=None,P=None):
        event = self.event if event is None else event
        if event in self.records:
            self.records[event]['target'][name] = data[0].to_json()
            self.records[event]['operation'][name] = data[1].to_json()
            self.records[event]['performance'][name] = data[2]
        elif event is None:
            pass
        else:
            self.records[event] = {'target':{name:data[0].to_json()},
            'operation':{name:data[1].to_json()},
            'performance':{name:data[2]}}
        if P is not None:
            self.records[event]['P'] = P
        self.event = event

    def save(self,model_dir=None):
        filedir = self.cwd if model_dir is None else model_dir
        if os.path.isdir(filedir):
            filedir = os.path.join(filedir,'records.json')
        with open(filedir,'w') as f:
            f.write(json.dumps(self.records))
    
    def load(self,model_dir=None):
        filedir = self.cwd if model_dir is None else model_dir
        if os.path.isdir(filedir):
            filedir = os.path.join(filedir,'records.json')
        with open(filedir,'r') as f:
            self.records = json.load(f)
        