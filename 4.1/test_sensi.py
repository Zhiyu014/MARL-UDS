# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:04:00 2021

@author: MOMO
"""

from env_swmm import Ast
from vdn import VDN
from qagent import QAgent
# import tqdm
from os import listdir
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook

import random
import pandas as pd
from swmm_api import read_out_file,read_rpt_file
from datetime import timedelta
from numpy import array,save,load
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999

env = Ast()
agents = []
for i in range(env.n_agents):
    agent = QAgent(epsilon_decay,env.action_size,env.observ_size,dueling=True)
    agents.append(agent)
vdnn = VDN(agents=agents)
vdnn.load()


def show_flooding(ax,labels,files,label,column,cumulative=False,label_position='right'):
    outfiles = [inpfile.replace('.inp','.out') for inpfile in files]
    floods = {}
    res = {}
    lines = []
    for idx,out_file in enumerate(outfiles):
        out = read_out_file(out_file)
        if type(label) is tuple:
            flood = out.get_part(*label)[column]
        else:
            flood = out.get_part(label)[column]
        floods[labels[idx]] = round(flood.sum(),3)
        
        if cumulative:
            flood = flood*5*60/1000
            flood = flood.cumsum()
            
        time = [ts.strftime('%H:%M') for ts in flood.index.tolist()]
        vol = flood.values
        line = ax.plot(time,vol,label = labels[idx])
        lines+=line
        ax.yaxis.set_ticks_position(label_position)
        ax.yaxis.set_label_position(label_position)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
        # ax.set_xlabel('Time (H:M)')
        
        # flood.plot(legend = True,
        #            ax = ax,
        #            label = labels[idx],
        #            xlabel = 'Time (hrs)',
        #            ylabel = 'Flooding ($\mathregular{10^3} \mathregular{m^3}$)')
        out.close()
        
        rpt = read_rpt_file(out_file.replace('.out','.rpt'))
        node_flood = rpt.node_flooding_summary
        node_inflow = rpt.node_inflow_summary['Total_Inflow_Volume_10^6 ltr'].loc[node_flood.index]
        if len(node_flood) == 0:
            res[labels[idx]] = 1
        else:
            # inflow = rpt.flow_routing_continuity['Dry Weather Inflow']['Volume_10^6 ltr'] + rpt.flow_routing_continuity['Wet Weather Inflow']['Volume_10^6 ltr']
            res[labels[idx]] = 1-sum(node_flood['Total_Flood_Volume_10^6 ltr']/node_inflow*node_flood['Hours_Flooded']/24)
    return floods,res,lines

def get_cum_flood(file):
    outfile = file.replace('.inp','.out')
    out = read_out_file(outfile)
    flood = out.get_part('system')['Flow_lost_to_flooding']
    flood = flood*5*60/1000
    flood = flood.cumsum()
    vol = flood.values
    return vol
        
def read_rainfile(path):
    with open(path,'r') as f:
        lines = f.readlines()
    lines = [[line.split()[1],eval(line.split()[-1])] for line in lines]
    return lines

raintests = {file.split('_')[0]:eval(file.split('_')[1]) for file in listdir('./test/rains/') if file.endswith('txt') and file.startswith('Rain')}
rains = {file.split('_')[0]:read_rainfile('./test/rains/'+file) for file in listdir('./test/rains/') if file.endswith('.txt') and file.startswith('Rain')}




# fig = plt.figure(figsize=(20,25),dpi = 1200)
# test_file = './test/sensi/Rain{0}_{1}_{2}.inp'
# columns = ['Uncontrolled','BC','EFD','VDN']
# for i in range(1,5):
#     rain_id = 'Rain %s'%i
#     rain = rains[rain_id]
#     files = ['./test/sensi/Rain%s_'%i+col+'.inp' for col in columns]
#     no_reward = env.test_no(rain,files[0])
#     bc_reward = env.test_bc(rain,files[1])
#     efd_reward = env.test_efd(rain,files[2])
#     test_reward,_,_ = env.test(vdnn,rain,test_inp_file = files[-1])
#     for sen in range(1,6):
#         ax = fig.add_subplot(5,4,(sen-1)*4+i)
#         _,_,lines = show_flooding(ax,columns,files,'system','Flow_lost_to_flooding',True,'left')
#         time = lines[0].get_xdata()
#         vols = []
#         for j in range(50):
#             if exists(test_file.format(i,sen,j).replace('.inp','.out')) == False:
#                 test_reward,_,_ = env.test(vdnn,rain,test_inp_file = test_file.format(i,sen,j),sensi = 0.1*sen)
#             vol = get_cum_flood(test_file.format(i,sen,j))
#             vols.append(vol)
#         vols = array(vols)
#         min_vol,max_vol = vols.min(axis=0)[:-1],vols.max(axis=0)[:-1]
#         fill = ax.fill_between(time,min_vol,max_vol,label = 'Uncertainty Buffer')
#         ax.legend(lines+[fill],[l.get_label() for l in lines+[fill]],loc='lower right')
#         if i == 1:
#             ax.set_ylabel('Cumulative CSO ($\mathregular{10^3} \mathregular{m^3}$)')
#         if sen == 1:
#             ax.set_title(rain_id)
#         if sen == 5:
#             ax.set_xlabel('Time (H:M)')
# fig.savefig('./test/results/sensi.png',dpi = 1200)
            
            
    
test_file = './test/sensi/Rain{0}_{1}_{2}.inp'
columns = ['Uncontrolled','BC','EFD','VDN']
datas = []
refers = []
for i in range(1,5):
    rain_id = 'Rain %s'%i
    rain = rains[rain_id]

    file_name = './test/'+rain_id+'_%s_'%i
    files = [file_name + alg + '.inp' for alg in columns]
    
    # no_reward = env.test_no(rain,files[0])
    # bc_reward = env.test_bc(rain,files[1])
    # efd_reward = env.test_efd(rain,files[2])
    # test_reward,_,_ = env.test(vdnn,rain,test_inp_file = files[-1])
    
    refers.append([read_rpt_file(file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr'] 
    for file in files])
    
    volss = []
    for sen in range(1,6):
        # ax = fig.add_subplot(5,5,(sen-1)*4+i)
        # _,_,lines = show_flooding(ax,columns,files,'system','Flow_lost_to_flooding',True,'left')
        # ts = lines[0].get_xdata()
        vols = []
        for j in range(50):
            sensi_file = test_file.format(i,sen,j)
            if exists(sensi_file.replace('.inp','.out')) == False:
                test_reward,_,_ = env.test(vdnn,rain,test_inp_file = test_file.format(i,sen,j),sensi = 0.1*sen)
            vol = read_rpt_file(sensi_file.replace('inp','rpt')).flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
            vols.append(vol) 
        volss.append(vols)
    data = array(volss)
    datas.append(data)

save('./test/sensi/sensi_data.npy',array(datas))
save('./test/sensi/sensi_refer.npy',array(refers))


# box plot of uncertainty results
datas = load('./test/sensi/sensi_data.npy')
refers = load('./test/sensi/sensi_refer.npy')
columns = ['BC','EFD','VDN']
fig = plt.figure(figsize=(25,5),dpi = 600)
for i,data in enumerate(datas):
    refer = refers[i]
    ax = fig.add_subplot(1,5,i+1)
    labels = list('12345')
    data = data.T
    stats = cbook.boxplot_stats(data, labels=labels, bootstrap=10000)
    vdn = ax.bxp(stats,patch_artist=True, showfliers=True)
    for patch in vdn['boxes']:
        patch.set_facecolor('salmon')   
    for med in vdn['medians']:
        med.set_color('black')    
    lines = []
    colors = ['orange','green','red']
    for j,ref in enumerate(refer[1:]):
        line = ax.axhline(y=ref,linestyle='--',label=columns[j],color=colors[j])
        lines.append(line)
    rain_id = 'Rain %s'%(i+1)
    ax.legend(lines,[l.get_label() for l in lines],loc='upper left')
    ax.set_title(rain_id)
    ax.set_xlabel('Uncertainty level')
    ax.set_ylabel('Accumulated CSO volume ($\mathregular{10^3} \mathregular{m^3}$)')
fig.savefig('./test/results/sensi.png',dpi=600)