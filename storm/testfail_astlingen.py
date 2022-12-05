from envs.astlingen import astlingen
from envs.utilities import generate_split_file
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
from swmm_api import read_inp_file
import yaml
import os
import numpy as np
import multiprocessing as mp
import random
from itertools import combinations_with_replacement

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

def observ_sensi_test(env,arg,event=None,observ_fail=False,items=[]):
    items = [ID for ID,_ in env.config['states'] if ID not in items]
    states = [ID for ID,_ in env.config['states']]
    items = [states.index(item) for item in items]

    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    state = env.reset(event)
    prev_state = state.copy()
    setting = bc_controller(env.config)
    done = False
    while not done:
        if observ_fail:
            state = [s if random.random()>observ_fail or idx in items else prev_state[idx]
            for idx,s in enumerate(state)]   
        action = f.act(state,False)
        setting = f.convert_action_to_setting(action)
        setting = np.array(setting).astype(float)
        done = env.step(setting)
        prev_state = state.copy()
        state = env.state()
    perf = env.performance('cumulative')
    return perf
    
def observ_attrs(env,arg,event=None,observ_fail={}):
    observ_fail = {idx:observ_fail[attr] if attr in observ_fail else 0 for idx,(_,attr) in enumerate(env.config['states'])}

    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    state = env.reset(event)
    prev_state = state.copy()
    setting = bc_controller(env.config)
    done = False
    while not done:
        if observ_fail:
            state = [s if random.random()>observ_fail[idx] else prev_state[idx]
            for idx,s in enumerate(state)]
        action = f.act(state,False)
        setting = f.convert_action_to_setting(action)
        setting = np.array(setting).astype(float)
        done = env.step(setting)
        prev_state = state.copy()
        state = env.state()
    perf = env.performance('cumulative')
    return perf

def interact_steps_fail(env,arg,event=None,sensor_fail=False,observ_fail=False,act_fail=False,backup=None):
    rb_items = []
    # rb_items = [item for items in env.config['site'].values()
    #  for item in items['states'] if item[0] in ['T','V']]
    states = [ID for ID,_ in env.config['states']]
    # rb_items = [states.index(item) for item in rb_items]

    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    if act_fail and type(backup) == type(arg):
        bk = backup.agent_class(backup.observ_space,backup.action_shape,backup)
    state = env.reset(event)
    prev_state = state.copy()
    setting = bc_controller(env.config)
    done = False
    while not done:
        if sensor_fail:
            state = [s if states[idx].startswith('V') else s*random.uniform(1-sensor_fail,1+sensor_fail)
            for idx,s in enumerate(state)]
        if observ_fail:
            if f.name in ['IQL','VDN']:
                # state = [s if random.random()>observ_fail or idx in rb_items else 0.0 for s in state]
                state = [s if random.random()>observ_fail or idx in rb_items else prev_state[idx]
                for idx,s in enumerate(state)]
            else:
                # state = [s*int(random.random()>observ_fail) for s in state]
                state = [s if random.random()>observ_fail else prev_state[idx]
                for idx,s in enumerate(state)]   
        action = f.act(state,False)
        if act_fail:
            if backup is None:
                setting = [sett if random.random()>act_fail else setting[idx]
                for idx,sett in enumerate(f.convert_action_to_setting(action))]
            elif backup is bc_controller:
                bc_setting = bc_controller(env.config)
                setting = [sett if random.random()>act_fail else bc_setting[idx]
                for idx,sett in enumerate(f.convert_action_to_setting(action))]
            elif type(backup) == type(arg):
                bk_setting = bk.convert_action_to_setting(bk.act(state,False))
                setting = [sett if random.random()>act_fail else bk_setting[idx]
                for idx,sett in enumerate(f.convert_action_to_setting(action))]
        else:
            setting = f.convert_action_to_setting(action)
        setting = np.array(setting).astype(float)
        done = env.step(setting)
        prev_state = state.copy()
        state = env.state()
    perf = env.performance('cumulative')
    return perf

def bc_controller(config):
    return [v[1] for v in config['settings'].values()]

def bc_test(env,event=None):
    _ = env.reset(event)
    setting = [v[1] for v in env.config['settings'].values()]
    done = False
    while not done:
        done = env.step(setting)
    perf = env.performance('cumulative')
    return perf

if __name__ == '__main__':
    # init SWMM environment and arguments
    env = astlingen(initialize=False)
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']

    # init control agents
    ctrls = {}
    for agent,item in hyp_test['test_agents'].items():
        hyp = hyps[agent]
        # update search parameters in the agent arg
        if hyp_test['if_predict']:
            hyp.update(hyps['predict'])
        env_args = env.get_args(if_mac=hyp['if_mac'])
        arg = Arguments(env_args,hyp)
        arg.init_before_testing(item)
        ctrls[agent] = arg
    ctrls['BC'] = bc_controller

    # init test args
    args = Arguments(env.get_args(), hyp_test)

    # generate rainfall
    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    test_events = generate_split_file(args.swmm_input,filedir=test_event_dir,rain_num=args.rainfall['test_events'],rain_arg=args.rainfall)



    if args.fail['sensor_fail']:
        sen_logger = args.init_test(columns=['sensor'],load=False)
        for idx,event in enumerate(test_events):
            rain_name = read_inp_file(event).OPTIONS['START_DATE'].strftime('%m/%d/%Y') + '-' + read_inp_file(event).OPTIONS['START_TIME'].strftime('%H')
            for agent,arg in ctrls.items():
                if agent == 'BC':
                    sen_logger.log((bc_test(env,event),),'BC',rain_name)
                    continue
                volss = []
                for f in range(11):           
                    if args.processes > 1:
                        pool = mp.Pool(args.processes)
                        res = [pool.apply_async(func=interact_steps_fail,args=(env,arg,event,0.1*f,False,False,None,))
                        for _ in range(args.fail['fail_num'])]
                        pool.close()
                        pool.join()
                        res = [r.get() for r in res]
                    else:
                        res = []
                        for _ in range(args.fail['fail_num']):
                            r = interact_steps_fail(env,arg,event,sensor_fail=0.1*f)
                            res.append(r)
                    volss.append(res)
                sen_logger.log((volss,),agent,rain_name)
            sen_logger.save(os.path.join(sen_logger.cwd,'sen_comm_fail.json'))


    if args.fail['obs_sensi']:
        obs_logger = args.init_test(columns=['rainfall','depthN'],load=False)
        for idx,event in enumerate(test_events):
            rain_name = read_inp_file(event).OPTIONS['START_DATE'].strftime('%m/%d/%Y') + '-' + read_inp_file(event).OPTIONS['START_TIME'].strftime('%H')
            for agent,arg in ctrls.items():
                if agent == 'BC':
                    continue
                agent_data = {}
                for col in obs_logger.columns:
                    items = [ID for ID,attr in env.config['states'] if col.startswith(attr)]
                    if col.startswith('depth'):
                        items = [item for item in items if item.startswith(col[-1])]

                    volss = []
                    for f in list(combinations_with_replacement(range(10),2)):
                        f = {'rainfall':f[0]*0.1,'depthN':f[1]*0.1}
                        if args.processes > 1:
                            pool = mp.Pool(args.processes)
                            res = [pool.apply_async(func=observ_attrs,args=(env,arg,event,f,))
                            for _ in range(args.fail['fail_num'])]
                            pool.close()
                            pool.join()
                            res = [r.get() for r in res]
                        else:
                            res = []
                            for _ in range(args.fail['fail_num']):
                                r = observ_attrs(env,arg,event,observ_fail=f,)
                                res.append(r)
                        volss.append(res)
                    agent_data[col] = volss
                obs_logger.log(list(agent_data.values()),agent,rain_name)
            obs_logger.save(os.path.join(obs_logger.cwd,'obs_sensi_item.json'))
            print('Finish observ: '+rain_name)

    if args.fail['obs_fail']:
        obs_logger = args.init_test(columns=['observ'],load=False)
        obs_logger.load(os.path.join(obs_logger.cwd,'obs_comm_fail_depth.json'))
        for idx,event in enumerate(test_events):
            rain_name = read_inp_file(event).OPTIONS['START_DATE'].strftime('%m/%d/%Y') + '-' + read_inp_file(event).OPTIONS['START_TIME'].strftime('%H')
            for agent,arg in ctrls.items():
                if agent == 'BC':
                    obs_logger.log((bc_test(env,event),),'BC',rain_name)
                    continue
                volss = []
                for f in range(11):           
                    if args.processes > 1:
                        pool = mp.Pool(args.processes)
                        res = [pool.apply_async(func=interact_steps_fail,args=(env,arg,event,False,0.1*f,False,None,))
                        for _ in range(args.fail['fail_num'])]
                        pool.close()
                        pool.join()
                        res = [r.get() for r in res]
                    else:
                        res = []
                        for _ in range(args.fail['fail_num']):
                            r = interact_steps_fail(env,arg,event,observ_fail=0.1*f)
                            res.append(r)
                    volss.append(res)
                obs_logger.log((volss,),agent,rain_name)
            obs_logger.save(os.path.join(obs_logger.cwd,'obs_comm_fail.json'))
            print('Finish observ: '+rain_name)

    if args.fail['act_fail']:
        act_logger = args.init_test(columns=['act'],load=False)
        for idx,event in enumerate(test_events):
            rain_name = read_inp_file(event).OPTIONS['START_DATE'].strftime('%m/%d/%Y') + '-' + read_inp_file(event).OPTIONS['START_TIME'].strftime('%H')
            cen_arg = ctrls['DQN']
            for agent in args.fail['backup']:
                arg = ctrls[agent] if agent in ctrls else None
                volss = []
                for f in [1,3,5,7,9]:     
                    if args.processes > 1:
                        pool = mp.Pool(args.processes)
                        res = [pool.apply_async(func=interact_steps_fail,args=(env,cen_arg,event,False,False,0.1*f,arg))
                        for _ in range(args.fail['fail_num'])]
                        pool.close()
                        pool.join()
                        res = [r.get() for r in res]
                    else:
                        res = []
                        for _ in range(args.fail['fail_num']):
                            r = interact_steps_fail(env,cen_arg,event,act_fail=0.1*f,backup=arg)
                            res.append(r)
                    volss.append(res)
                act_logger.log((volss,),'DQN&'+str(agent),rain_name)
            act_logger.save(os.path.join(act_logger.cwd,'act_comm_fail.json'))
            print('Finish act: '+rain_name)

