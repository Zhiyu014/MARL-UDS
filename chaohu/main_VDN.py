# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:58:18 2021

@author: MOMO
"""
from envs.chaohu import chaohu
from envs.utilities import generate_file,eval_control
from agent.vdn import VDN
from utils.memory import RandomMemory
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
from utils.logger import logger
import yaml
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

def interact_steps(env,f,event=None,train=True,base=None):
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
        reward = env.reward(done,base)
        rewards += reward
        traj += [action,reward,state,float(done)]
        trajs.append(traj)
    perf = env.performance('cumulative')
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
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)
    env = chaohu()
    env_args = env.get_args(if_mac=hyps[env.config['env_name']]['VDN']['if_mac'])
    args = Arguments(env_args,hyp=hyps[env_args['env_name']]['VDN'])
    args.init_before_training()

    memory = RandomMemory(args.max_capacity)
    ctrl = VDN(args.observ_space,args.action_space,args)
    # ctrl.load(norm=True)

    train_event_dir = os.path.splitext(args.swmm_input)[0] + '_train.inp'
    train_events = generate_file(args.swmm_input,args.rainpara,
                                 filedir=train_event_dir,
                                 rain_num=args.explore_events,
                                 replace=False)

    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    test_events = generate_file(args.swmm_input,args.rainpara,
                                 filedir=test_event_dir,
                                 rain_num=args.eval_events,
                                 replace=False)

    hc_trains = [hc_test(env,event) for event in train_events]
    hc_tests = [hc_test(env,event) for event in test_events]

    log = logger(args.cwd)
    for n in range(args.total_episodes + args.pre_episodes):
        # Sampling
        rewards = []
        for idx,event in enumerate(train_events):
            trajs,reward,perf = interact_steps(env,ctrl,event,hc_trains[idx])
            memory.update(trajs)
            rewards.append(reward)
            print('Training Reward at event {0}: {1}'.format(idx,reward))
            print('Training Score at event {0}: {1}'.format(idx,perf))
            print('HC Score: %s'%hc_trains[idx])
        print('Sampling Complete: %s'%n)

        # Training
        if n >= args.pre_episodes:
            print('Upgrading')
            losses = ctrl.update_net(memory)
            print('Upgrade Complete: %s'%n)
            update = log.log((rewards,losses),train=True)
            if update:
                ctrl.save(os.path.join(ctrl.model_dir,'train'))

        # Evaluate the model in several episodes
        if n % args.eval_gap == 0 and n > args.pre_episodes:
            perfs = []
            losses = []
            for idx,event in enumerate(test_events):
                trajs,_,perf = interact_steps(env,ctrl,event,train=False,base=hc_tests[idx])
                loss = ctrl.evaluate_net(trajs)
                perfs.append(perf)
                losses.append(loss)
                print('Testing Score at event {0}: {1}'.format(idx,perf))
                print('HC Score: %s'%hc_tests[idx])
            update = log.log((perfs,losses),train=False)
            if update:
                ctrl.save(os.path.join(ctrl.model_dir,'test'))

        # Save the current model
        if n%args.save_gap == 0:
            ctrl.save()

    log.save()
    log.plot()