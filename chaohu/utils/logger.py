import numpy as np
from functools import reduce
import os
from matplotlib import pyplot as plt
plt.rc('font',family = 'Times New Roman')

class logger():
    def __init__(self,model_dir):
        self.attrs = ['{0}_{1}'.format(k,v)
                         for k in ['train','test'] 
                         for v in ['perfs','loss']]
        for attr in self.attrs:
            setattr(self,attr,list())

        self.min_train_perf = 1e10
        self.min_test_perf = 1e10

        self.cwd = model_dir

    def log(self,data,train=False):
        if train:
            self.train_perfs.append(data[0])
            self.train_loss.append(data[1])
            update =  sum(data[0]) < self.min_train_perf
            self.min_train_perf = min(self.min_train_perf,sum(data[0]))
        else:
            self.test_perfs.append(data[0])
            self.test_loss.append(data[1])
            update =  sum(data[0]) < self.min_test_perf
            self.min_test_perf = min(self.min_test_perf,sum(data[0]))
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

    def plot(self,filedir=None,event_wise=False):
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
        axP.plot(np.arange(len(train_loss)),train_loss,label='loss')
        axP.set_xlabel('Update times')
        axP.set_title("training loss")
        axP.legend(loc='upper right')


        if event_wise:
            for idx in range(len(self.test_perfs[0])):
                perf = [per[idx] for per in self.test_perfs]
                axM.plot(np.arange(len(perf)),perf,label = 'event_%s'%idx)
        else:
            test_perfs = np.array(self.test_perfs).sum(axis=1)
            axM.plot(np.arange(len(test_perfs)),test_perfs,label = 'perf')
        axM.set_xlabel('episode')
        axM.set_title('test performance')
        axM.legend(loc='lower right')


        if event_wise:
            for idx in range(len(self.test_loss[0])):
                loss = [los[idx] for los in self.test_loss]
                axR.plot(np.arange(len(loss)),loss,label = 'event_%s'%idx)
        else:
            test_loss = np.array(self.test_loss).sum(axis=1)
            axR.plot(np.arange(len(test_loss)),test_loss,label = 'loss')
        axR.set_xlabel('episode')
        axR.set_title("test loss")
        axR.legend(loc='upper right')

        if filedir is None:
            fig.savefig(os.path.join(self.cwd,'training.png'))
        elif os.path.isdir(filedir):
            fig.savefig(os.path.join(filedir,'training.png'))
        else:
            fig.savefig(filedir)





