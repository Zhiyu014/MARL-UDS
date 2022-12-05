# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:58:18 2021

@author: MOMO
"""
from envs.RedChicoSur import RedChicoSur
from utils.memory import RandomMemory
# from rnmemory import Recurrent_RandomMemory
import yaml
import os
import numpy as np
import multiprocessing as mp
# import multiprocess as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.config import Arguments
from functools import reduce
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

def interact_steps(env,arg,train=True):
    if type(arg) is Arguments:
        f = arg.agent_class(arg.observ_space,arg.action_shape,arg,act_only=True)
    else:
        f = arg
    trajs = []
    state = env.reset()
    done = False
    rewards = []
    while not done:
        traj = [state]
        action = f.act(state,train)
        setting = f.convert_action_to_setting(action)
        done = env.step(setting,env.config['control_interval']*60)
        state = env.state()
        # Use designed reward function
        reward = env.reward(ctrl.global_reward)/1000
        rewards.append(reward)
        traj += [action,reward,state,done]
        trajs.append(traj)
    perf = env.performance('cumulative')[-1]
    reward = np.array(rewards).sum(axis=0).mean()
    if train:
        print('Reward: %s'%reward)
    print('Score: %s'%perf)
    return trajs,reward,perf

def bc_test(env):
    _ = env.reset()
    done = False
    while not done:
        done = env.step(advance_seconds=env.config['control_interval']*60)
    perf = env.performance('cumulative')[-1]
    return perf

if __name__ == '__main__':
    # init SWMM environment and arguments
    env = RedChicoSur(initialize=False)
    env.config['swmm_input'] = os.path.join(os.path.dirname(env.config['swmm_input']),'test.inp')
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)
    hyp = hyps[env.config['env_name']]
    hyp = hyp[hyp['train']]

    env_args = env.get_args(if_mac=hyp['if_mac'])
    args = Arguments(env_args,hyp=hyp)

    log = args.init_before_training()

    memory = RandomMemory(args.max_capacity, args.cwd, args.if_load)
    ctrl = args.agent_class(args.observ_space,args.action_shape,args)

    # BC tests
    bc_trains = bc_test(env)

    #print('HC tests complete')
    # ini_n = getattr(args,'ini_episodes',0)
    # for n in range(ini_n, args.total_episodes + args.pre_episodes):

    while args.episode <= args.total_episodes + args.pre_episodes:

        res = interact_steps(env,ctrl)
        trajs,rewards,perfs = [[res[i]] for i in range(3)]
        memory.update(trajs[0])
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

        # Evaluate the model in several episodes
        if args.episode % args.eval_gap == 0:
            trajs,_,perf = interact_steps(env,ctrl,train=False)
            losses = ctrl.evaluate_net(trajs)
            update = log.log(([perf],[losses]),train=False)
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
