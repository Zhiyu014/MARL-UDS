# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:27:38 2021

@author: chong
"""
from env_swmm import Ast
from vdn import VDN
from qagent import QAgent
# import tqdm
from os import listdir
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from swmm_api import read_out_file,read_rpt_file
from datetime import timedelta
plt.rc('font',family = 'Times New Roman')
epsilon_decay = 0.999	


# init SWMM environment and agents
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
        
        if cumulative:
            flood = flood.cumsum()
            
        time = [ts.strftime('%H:%M') for ts in flood.index.tolist()]
        vol = flood.values
        line = ax.plot(time,vol,label = labels[idx])
        lines+=line
        ax.yaxis.set_ticks_position(label_position)
        ax.yaxis.set_label_position(label_position)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
        ax.set_xlabel('Time (H:M)')
        
        # flood.plot(legend = True,
        #            ax = ax,
        #            label = labels[idx],
        #            xlabel = 'Time (hrs)',
        #            ylabel = 'Flooding ($\mathregular{10^3} \mathregular{m^3}$)')
        out.close()
        
        rpt = read_rpt_file(out_file.replace('.out','.rpt'))
        floods[labels[idx]] = rpt.flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
        node_flood = rpt.node_flooding_summary
        node_inflow = rpt.node_inflow_summary['Total_Inflow_Volume_10^6 ltr'].loc[node_flood.index]
        if len(node_flood) == 0:
            res[labels[idx]] = 1
        else:
            # inflow = rpt.flow_routing_continuity['Dry Weather Inflow']['Volume_10^6 ltr'] + rpt.flow_routing_continuity['Wet Weather Inflow']['Volume_10^6 ltr']
            res[labels[idx]] = 1-sum(node_flood['Total_Flood_Volume_10^6 ltr']/node_inflow*node_flood['Hours_Flooded']/24)
    floods['Precipitation'] = rpt.runoff_quantity_continuity['Total Precipitation']['Depth_mm']
    return floods,res,lines



def read_rainfile(path):
    with open(path,'r') as f:
        lines = f.readlines()
    lines = [[line.split()[1],eval(line.split()[-1])] for line in lines]
    return lines

def write_rainfile(rain,rain_id,P):
    rain_name = rain_id+'_{0}_'.format(P)
    path = './test/rains/'+rain_name+'.txt'
    if exists(path) == False:
        with open(path,'w') as f:
            time = [ra[0] for ra in rain]
            intensity = [ra[1] for ra in rain]
            lines = ['01/01/2000  '+ti+'  '+str(inte)+'\n' for ti,inte in zip(time,intensity)]
            f.writelines(lines)
            

raintests = {file.split('_')[0]:eval(file.split('_')[1]) for file in listdir('./test/rains/') if file.endswith('txt') and file.startswith('Rain')}
if len(raintests) == 0:
    # raintests = {'Rain %s'%(idx+1):random.randint(1,10) for idx in range(9)}
    raintests = {'Rain %s'%(idx+1):idx+1 for idx in range(8)}
    rains = {rain_id:env.euler_II_Hyeto(None, P) for rain_id,P in raintests.items()}
    for rain_id,P in raintests.items():
        write_rainfile(rains[rain_id],rain_id,P)
else:
    rains = {file.split('_')[0]:read_rainfile('./test/rains/'+file) for file in listdir('./test/rains/') if file.endswith('.txt') and file.startswith('Rain')}


columns = ['Uncontrolled','BC','EFD','VDN']
test_floodings = pd.DataFrame(columns = ['Precipitation']+columns,index=rains.keys())
test_res = pd.DataFrame(columns = ['Precipitation']+columns,index=rains.keys())

fig = plt.figure(figsize=(15,20),dpi = 600)
plt.tight_layout(pad=0.4,w_pad=0.5)
# fig2 = plt.figure(figsize = (10,5))
# ax2 = fig2.add_subplot(1,1,1)
for rain_id,rain in rains.items():
    idx = eval(rain_id.split()[-1])
    P = raintests[rain_id]
    time = [ra[0] for ra in rain]+[str(t//60).zfill(2)+':'+str(t % 60).zfill(2) for t in range(125,240,5)]
    intensity = [ra[1] for ra in rain]+[0]*23

    
    rain_name = rain_id+'_{0}_'.format(P)
    # with open('./test/rains/'+rain_name+'.txt','w') as f:
    #     lines = ['01/01/2000  '+ti+'  '+str(inte)+'\n' for ti,inte in zip(time,intensity)]
    #     f.writelines(lines)
    
    file_name = './test/'+rain_name
    files = [file_name + alg + '.inp' for alg in columns]
    test_reward,acts,_ = env.test(vdnn,rain,test_inp_file = files[-1])
    no_reward = env.test_no(rain,files[0])
    bc_reward = env.test_bc(rain,files[1])
    efd_reward = env.test_efd(rain,files[2])
    
    ax = fig.add_subplot(4,2,idx)
    ax.set_title(rain_id)
    
    
    ax2 = ax.twinx()
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
    ax2.invert_yaxis()
    ax2.set_ylim((34,0))
    bar = ax2.bar(time,intensity,label='Rainfall',alpha=0.7)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(6))
    ax2.set_xlabel('Time (hrs)')
    ax2.set_ylabel('Rainfall Intensity (mm/hr)',labelpad=5.0)
    
    floods,res,lines = show_flooding(ax,labels=columns,files = files,label='system',column='Flow_lost_to_flooding',cumulative=False)
    objs = lines + [bar]
    labs = [l.get_label() for l in objs]
    ax.set_ylabel('CSO ($\mathregular{m^3/s}$)',labelpad=5)
    ax2.legend(objs,labs,loc = 'upper right')

    test_floodings.loc[rain_id,:] = floods
    test_res.loc[rain_id,:] = res
    
fig.savefig('./test/flooding/results2.png')
test_floodings = test_floodings.astype(float).round(3)
test_res = test_res.astype(float).round(3)


# plot tank depths
columns = ['Uncontrolled','BC','EFD','VDN']
rain_name = 'Rain 3'+'_{0}_'.format(3)    
file_name = './test/'+rain_name
files = [file_name + alg + '.out' for alg in columns]
fig = plt.figure(figsize=(15,10),dpi = 600)
for idx,i in enumerate([2,3,4,6]):
    ax = fig.add_subplot(2,2,idx+1)
    ax.set_title('T%s'%i)
    lines = []
    for idx,outfile in enumerate(files):
        out = read_out_file(outfile)
        depth = out.get_part('node','T%s'%i)['Depth_above_invert']
        out.close()
        time = [ti.strftime('%H:%M') for ti in depth.index.tolist()]
        dep = depth.values.tolist()
        line = ax.plot(time,dep,label = columns[idx])
        lines += line
    ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
    ax.legend(lines,columns,loc='lower right')
    ax.set_ylabel('Depth ($m$)',labelpad=5)
    
fig.savefig('./test/results/tank_depth.png')      

# plot orifice settings
columns = ['Uncontrolled','BC','EFD','VDN']
rain_name = 'Rain 3'+'_{0}_'.format(3)
file_name = './test/'+rain_name
files = [file_name + alg + '.out' for alg in columns]
fig2 = plt.figure(figsize=(15,10),dpi = 600)
for idx,i in enumerate([2,3,4,6]):
    ax2 = fig2.add_subplot(2,2,idx+1)
    ax2.set_title('V%s'%i)
    lines2 = []
    for idx,outfile in enumerate(files):
        out = read_out_file(outfile)
        setting = out.get_part('link','V%s'%i)['Capacity']
        out.close()
        time = setting.index.tolist()
        time = [t+timedelta(minutes=i) for t in time for i in range(5)]
        time = [t.strftime('%H:%M') for t in time]
        capa = [ca for ca in setting.values.tolist() for i in range(5)]
        line = ax2.plot(time,capa,label = columns[idx])
        lines2+=line
    ax2.set_ylim((-0.1,1.1))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax2.set_xlabel('Time (H:M)')
    ax2.legend(lines2,columns,loc='lower right')
    ax2.set_ylabel('Setting',labelpad=5)
fig2.savefig('./test/results/orifice_setting.png')  



