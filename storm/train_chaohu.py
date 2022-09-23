# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:58:18 2021

@author: MOMO
"""
from envs.chaohu import chaohu
from envs.utilities import generate_file,eval_control
from utils.memory import RandomMemory
# from rnmemory import Recurrent_RandomMemory
import yaml
import os
import multiprocessing as mp
# import multiprocess as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.config import Arguments
from functools import reduce
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

def interact_steps(env,arg,event=None,train=True,base=None):
    if type(arg) is Arguments:
        f = arg.agent_class(arg.observ_space,arg.action_shape,arg,act_only=True)
    else:
        f = arg
    trajs = []
    state = env.reset(event)
    done = False
    rewards = 0
    while not done:
        traj = [state]
        action = f.act(state,train)
        setting = f.convert_action_to_setting(action)
        done = env.step(setting,env.config['control_interval']*60)
        state = env.state()
        # Use designed reward function
        reward = env.reward(done,base)
        # Use the basic reward function (related to CSO & flooding)
        # reward = -0.001 * env.performance()
        rewards += reward
        traj += [action,reward,state,done]
        trajs.append(traj)
    perf = env.performance('cumulative')
    
    if train:
        print('Training Reward at event {0}: {1}'.format(os.path.basename(event),rewards))
        print('Training Score at event {0}: {1}'.format(os.path.basename(event),perf))
        print('HC Score: %s'%base)
    else:
        print('Evaluation Score at event {0}: {1}'.format(os.path.basename(event),perf))
        print('HC Score: %s'%base)
    return trajs,rewards,perf

def hc_test(env,event=None):
    def hc_controller(depth,setting):
        starts = [int(depth[0]>h) for h in [0.8,1,1.2,1.4]] + [int(depth[1]>h) for h in [4,4.2,4.3]]
        shuts = [1-int(depth[0]<0.5) for _ in range(4)] + [1-int(depth[1]<h) for h in [1,1,2,1.2]]
        setting = [max(sett,starts[i]) for i,sett in enumerate(setting)]
        setting = [min(sett,shuts[i]) for i,sett in enumerate(setting)]
        return setting

    _ = env.reset(event)
    setting = [0 for _ in env.config['action_space']]
    done = False
    while not done:
        depth = env.env._state()[1:3]
        setting = hc_controller(depth,setting)
        done = env.step(setting,env.config['control_interval']*60)
    perf = env.performance('cumulative')
    return perf

if __name__ == '__main__':
    # init SWMM environment and arguments
    env = chaohu(initialize=False)
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)
    hyp = hyps[env.config['env_name']]
    hyp = hyp[hyp['train']]

    env_args = env.get_args(if_mac=hyp['if_mac'])
    args = Arguments(env_args,hyp=hyp)

    log = args.init_before_training()

    memory = RandomMemory(args.max_capacity, args.cwd, args.if_load)
    ctrl = args.agent_class(args.observ_space,args.action_shape,args)

    train_event_dir = os.path.splitext(args.swmm_input)[0] + '_train.inp'
    train_events = generate_file(args.swmm_input,args.rainfall_parameters,
                                 filedir=train_event_dir,
                                 rain_num=args.explore_events,
                                 replace=args.replace_rain)

    eval_event_dir = os.path.splitext(args.swmm_input)[0] + '_eval.inp'
    eval_events = generate_file(args.swmm_input,args.rainfall_parameters,
                                 filedir=eval_event_dir,
                                 rain_num=args.eval_events,
                                 replace=args.replace_rain)

    # HC tests
    if args.processes > 1:
       pool = mp.Pool(args.processes)
       res = [pool.apply_async(func=hc_test,args=(env,event,)) for event in train_events+eval_events]
       pool.close()
       pool.join()
       hc_trains = [r.get() for r in res[:len(train_events)]]
       hc_evals = [r.get() for r in res[len(train_events):]]
    else:
       hc_trains = [hc_test(env,event) for event in train_events]
       hc_evals = [hc_test(env,event) for event in eval_events]

    #print('HC tests complete')
    # ini_n = getattr(args,'ini_episodes',0)
    # for n in range(ini_n, args.total_episodes + args.pre_episodes):

    while args.episode <= args.total_episodes + args.pre_episodes:
        # Sampling TODO: Parallel Problems in Linux
        # pool must be initialzed before retrieving data
        if args.processes > 1:
            pool = mp.Pool(args.processes)
            res = []
            for idx,event in enumerate(train_events):
                r = pool.apply_async(func=interact_steps,args=(env,args,event,True,hc_trains[idx],))
                res.append(r)
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [interact_steps(env,ctrl,event,base=hc_trains[idx])
             for idx,event in enumerate(train_events)]
        trajs,rewards,perfs = [[r[i] for r in res] for i in range(3)]
        trajs = reduce(lambda x,y:x+y, trajs)
        memory.update(trajs)
        print('Sampling Complete: %s'%args.episode)
        
        if args.episode < args.pre_episodes:
            ctrl.episode_update(*args.episode_update())
            continue

        # Training
        print('Upgrading')
        losses = ctrl.update_net(memory)
        print('Upgrade Complete: %s'%args.episode)
        update = log.log((rewards,perfs,losses),train=True)
        if update[0]:
            ctrl.save(os.path.join(ctrl.model_dir,'train'))
        if update[1]:
            ctrl.save(os.path.join(ctrl.model_dir,'reward'))
        # load & save in each episode after pre_episodes
        if args.processes > 1:
            args.if_load = True
            ctrl.save()

        # Evaluate the model in several episodes
        if args.episode % args.eval_gap == 0:
            perfs = []
            losses = []
            for idx,event in enumerate(eval_events):
                trajs,_,perf = interact_steps(env,ctrl,event,train=False,base=hc_evals[idx])
                loss = ctrl.evaluate_net(trajs)
                perfs.append(perf)
                losses.append(loss)
            update = log.log((perfs,losses),train=False)
            if update:
                ctrl.save(os.path.join(ctrl.model_dir,'eval'))

        # Save the current model
        if args.episode % args.save_gap == 0:
            ctrl.save()
            memory.save()
            log.save()
            log.plot()

        # Update the episode and exploration greedy
        ctrl.episode_update(*args.episode_update())

    ctrl.save()
    memory.save()
    log.save()
    log.plot()
