
import os,yaml
import pandas as pd
import numpy as np
from envs.chaohu import chaohu
from utils.config import Arguments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('font',family = 'Times New Roman')

HERE = os.path.dirname(__file__)

if __name__ == '__main__':
    env = chaohu()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test(True)
    # logger.load(os.path.join(logger.cwd,'test_records.json'))

    pump_class = ['CC-S','CC-R','JK-S','JK-R']
    colors = {'IQL':'#1f77b4','VDN':'#2ca02c', 'DQN':'#ff7f0e','HC':'#d62728'}

    for event,record in logger.records.items():
        targets = {k:pd.read_json(v) for k,v in record['target'].items()}
        operats = {k:pd.read_json(v) for k,v in record['operation'].items()}

        # Plot CSO & flooding
        fig,(axL,axR) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
        for col,ax in zip(['Flooding','CSO'],[axL,axR]):
            rain = ax.bar(targets['HC'].index,targets['HC']['Rainfall'],label='rainfall',width=0.001,alpha=0.5,zorder=1)

            ax2 = ax.twinx()
            ax.invert_yaxis()
            objs = [rain]
            for agent,table in targets.items():
                flood = ax2.plot(table.index,table[col],color=colors[agent],label=agent)
                objs += flood
            ax2.set_xlabel('Time (H:M)')
            ax2.set_title(col)
            ax2.yaxis.set_ticks_position('right')
            ax2.yaxis.set_label_position('right')    
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # ax2.legend(objs,[l.get_label() for l in objs],loc='lower right')
        ax2.set_ylabel('Flooding/CSO ($\mathregular{10^6 L}$)')
        axL.set_ylabel('Rainfall Intensity (mm/h)')
        fig.legend(objs,[l.get_label() for l in objs],loc=8,ncol=5,frameon=False)
        fig.savefig(os.path.join(args.cwd,'perf_%s'%event))
        print("Finish plot: perf_%s"%event)
        
        # Plot the opertion details in CC & JK
        fig,axes = plt.subplots(nrows=len(operats),ncols=2,figsize=(10,5*(len(operats))))
        axes = np.array([axes]) if len(axes.shape) == 1 else axes
        for (agent,operat),(axL,axR) in zip(operats.items(),axes):

            for pump in pump_class:
                operat[pump] = operat.apply(lambda row:sum([row[col]
                for col in operat.columns if col.startswith(pump)]),axis=1).astype(int)
                operat[pump] = operat.apply(lambda row: 0 if row[pump[:3]+'storage']==0 else row[pump],axis=1)
            operat = operat[args.storage + pump_class]

            # rain = axL.bar(operat.index,operat['rainfall'],label='rainfall',width=0.003,alpha=0.6,zorder=1)
            # axL.set_ylabel('Rainfall Intensity (mm/h)')
            for col,ax in zip(['CC','JK'],[axL,axR]):
                depth = ax.plot(operat.index,operat[col+'-storage'],label='Storage Tank')
                ax2 = ax.twinx()
                ns = ax2.plot(operat.index,operat[col+'-S'],'--',label='Sewage pumps')
                nr = ax2.plot(operat.index,operat[col+'-R'],'--',label='Storm pumps')
                objs = depth+ns+nr
                ax2.set_xlabel('Time (H:M)')
                
                ax2.set_title(agent + '-'+ col)
                ax2.yaxis.set_ticks_position('right')
                ax2.yaxis.set_label_position('right')    
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.set_ylim(-0.3,env.node_properties[col+'-storage']['fullDepth']+0.2)

            ax2.set_ylabel('Number of pumps on')
            axL.set_ylabel('Depth (m)')
        fig.legend(objs,[l.get_label() for l in objs],loc=8,ncol=3,frameon=False)
        fig.savefig(os.path.join(args.cwd,'depth_setting_%s'%event))
        print("Finish plot: depth_setting_%s"%event)




