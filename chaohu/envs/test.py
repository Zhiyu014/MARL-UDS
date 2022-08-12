# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:24:55 2022

@author: MOMO
"""
import os
import multiprocessing as mp
import time
from chaohu import chaohu
from utilities import generate_file

HERE = os.path.dirname(__file__)
# Test if the chaohu env works well
# env = chaohu()
# done = False
# while not done:
#     state = env.state()
#     print(state[-4:])
#     done = env.step(actions = [1 for pump in env.config['action_space']],
#                     advance_seconds=env.config["control_interval"]*60)
#     perf = env.performance()

def run(file):
    env = chaohu(swmm_file = file)
    done = False
    while not done:
        state = env.state()
        done = env.step(actions = [1 for pump in env.config['action_space']],
                        advance_seconds=env.config["control_interval"]*60)
        perf = env.performance()

if __name__ == '__main__':
    env = chaohu()
    rainpara_file = os.path.join(HERE,'config','rainparas.yaml')
    swmm_filedir = os.path.join(HERE,'network','test_%s.inp')
    files = generate_file(env.config['swmm_input'],rainpara_file,rain_num=10,filedir=swmm_filedir)
    t0 = time.time()
    for file in files:
        run(file)
    # Lack of CPU resources makes multi-processing much slower
    # process = [mp.Process(target=run,args=(files[i],)) for i in range(10)]
    # for p in process:
    #     p.start()
    # process[-1].join()
    print(time.time()-t0)


# Tests for reward function
import os
from pyswmm import Simulation,Nodes,Links
sim = Simulation(os.path.join(os.getcwd(),'env','network','chaohu.inp'))
nodes = Nodes(sim)
links = Links(sim)
per = 0
ene = 0
cum_precip = 0
pumps = ['CC-S1','CC-S2','CC-R1','CC-R2','JK-S','JK-R1','JK-R2']
tanks = ['CC-storage','JK-storage']
for _ in sim:
    for pump in pumps:
        links[pump].target_setting=1
    sim.step_advance(300)
    ene_t = sum([links[pump].pump_statistics['energy_consumed'] for pump in pumps])
    # print(ene_t - ene)
    precip = sim._model.runoff_routing_stats()['rainfall']
    if precip - cum_precip == 0.0:
        reward = sum([1-nodes[tank].depth/nodes[tank].initial_depth for tank in tanks])
    else:
        reward = sum([nodes[tank].depth/nodes[tank].initial_depth-1 for tank in tanks])
    reward -= (ene_t - ene)/5
    cum_precip = precip
    ene = ene_t
    print(reward)

    # print(nodes['JK-2'].outfall_statistics['total_periods'] - per)
    # per = nodes['JK-2'].outfall_statistics['total_periods']

'''
Reward shaping:
1. Operational reward: ±2
    1) If raining:  sum[(depth - ini_depth)/ini_depth] - sum[energy] * 0.2
    2) Not raining: sum[(ini_depth - depth)/ini_depth] - sum[energy] * 0.2
2. Closing reward:  ±5 (gamma=0.95 n_steps=36 reward=2/(0.95**36)=4.16 --> 5)
    1) - 8 * (sum[flooding] + sum[overflow])/(sum[outflow] + sum[flooding] + final_storage) + 5
        cannot reach ±5 at all times so multiply -20 plus 15
    2) 10 * (HC[fl&CSO] - fl&CSO)/HC[fl&SCO]
        cannot reach ±5 at all times so multiply 10



'''

from swmm_api import read_rpt_file

def eval_closing_reward(rpt_file):
    rpt = read_rpt_file(rpt_file)
    routing = rpt.flow_routing_continuity
    outload = rpt.outfall_loading_summary
    A,B = -20,10
    return A*(routing['Flooding Loss']['Volume_10^6 ltr'] +\
         routing['External Outflow']['Volume_10^6 ltr'] -\
             outload.loc['WSC','Total_Volume_10^6 ltr'] )/\
                sum([routing[col]['Volume_10^6 ltr']
                 for col in routing.keys() if 'Inflow' in col or 'Initial' in col]) + B

for file in os.listdir():
    if file.endswith('.rpt') and file.startswith('chaohu_train'):
        print(eval_closing_reward(file))



# Test Avg_volume for storage
# Not suitable with inconsistence
import os
from pyswmm import Simulation,Nodes,Links
sim = Simulation(os.path.join(os.getcwd(),'env','network','chaohu.inp'))
nodes = Nodes(sim)
links = Links(sim)
st = sim.start_time
times = []
vols = []
for _ in sim:
    sim.step_advance(600)
    avg_vol = nodes['CC-storage'].statistics['average_depth']
    if len(vols)>0:
        vol = (avg_vol * (sim.current_time - st).total_seconds() - vols[-1] * (times[-1] - st).total_seconds())/(sim.current_time-times[-1]).total_seconds()
    else:
        vol = avg_vol
    times.append(sim.current_time)
    vols.append(avg_vol)
    print(avg_vol)