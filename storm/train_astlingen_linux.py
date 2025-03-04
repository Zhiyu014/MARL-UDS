# -*- coding: utf-8 -*-

import os,yaml
import multiprocessing as mp
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
# tf.config.list_physical_devices(device_type='GPU')

from envs.astlingen import astlingen
from envs.utilities import generate_split_file
from utils.memory import RandomMemory
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
from functools import reduce
import pandas as pd
HERE = os.path.dirname(__file__)


def interact_steps(env,arg,event=None,train=True,on_policy=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    if type(arg) is Arguments:
        with tf.device("/cpu:0"):
            f = arg.agent_class(arg.observ_space,arg.action_shape,arg,act_only=True)
    else:
        f = arg
    trajs = []
    state = env.reset(event,env.global_state,arg.seq_len if arg.if_recurrent else False)
    done = False
    rewards = 0
    while not done:
        traj = [state]
        action = f.act(state,train)
        if on_policy:
            action,log_probs = action
        setting = f.convert_action_to_setting(action)
        done = env.step(setting,env.config['control_interval']*60)
        state = env.state(arg.seq_len if arg.if_recurrent else False)
        reward = env.reward(norm = True)
        rewards += reward
        traj += [action,reward,state,done]
        if on_policy:
            value = f.criticize(traj[0])
            traj += [log_probs,value]
        trajs.append(traj)
    perf = env.performance('cumulative')
    
    if train:
        print('Training Reward at event {0}: {1}'.format(os.path.basename(event),rewards))
        print('Training Score at event {0}: {1}'.format(os.path.basename(event),perf))
    else:
        print('Evaluation Score at event {0}: {1}'.format(os.path.basename(event),perf))
    return trajs,rewards,perf

def bc_test(env,event=None):
    _ = env.reset(event)
    setting = [v[1] for v in env.config['settings'].values()]
    done = False
    while not done:
        done = env.step(setting)
    perf = env.performance('cumulative')
    return perf

def efd_test(env,event=None):
    def efd_controller(depth):
        asp = env.config['settings']
        setting = {k:1 for k in asp}
        if max(depth.values())<1:
            setting = {k:1 for k in asp}
        for k in asp:
            t = k.replace('V','T')
            setting[k] = 2 * int(depth[t] >= max(depth.values())) +\
                0 * int(depth[t] <= min(depth.values())) +\
                    1 * (1-int(depth[t] >= max(depth.values()))) * (1-int(depth[t] <= min(depth.values())))
        setting = [v[setting[k]] for k,v in asp['settings'].items()]
        return setting

    _ = env.reset(event)
    depth_index = {k:i for i,(k,v) in enumerate(env.config['states']) if v == 'depthN'}
    done = False
    while not done:
        depth = env.state()
        depth = {k:depth[i] for k,i in depth_index.items()}
        setting = efd_controller(depth)
        done = env.step(setting)
    perf = env.performance('cumulative')
    return perf


def update_agent(args,memory,log,res):
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
    import tensorflow as tf
    tf.config.list_physical_devices(device_type='GPU')
    with tf.device('/gpu:0'):
        ctrl = args.agent_class(args.observ_space,args.action_shape,args,False)
    losses = ctrl.update_net(memory)

    rewards,perfs = [[r[i] for r in res] for i in range(1,3)]
    update = log.log((rewards,perfs,losses),train=True)
    if update[0]:
        ctrl.save(os.path.join(ctrl.model_dir,'train'))
    if update[1]:
        ctrl.save(os.path.join(ctrl.model_dir,'reward'))
    # load & save in each episode after pre_episodes
    ctrl.save()
    return log

def evaluate_agent(args,log,res):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    with tf.device('/cpu:0'):
        ctrl = args.agent_class(args.observ_space,args.action_shape,args)
    perfs = [r[-1] for r in res]
    losses = [ctrl.evaluate_net(r[0]) for r in res]
    update = log.log((perfs,losses),train=False)
    if update:
        ctrl.save(os.path.join(ctrl.model_dir,'eval'))
    return log


if __name__ == '__main__':
    env = astlingen(initialize=False)
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)
    hyp = hyps[env.config['env_name']]
    hyp = hyp[hyp['train']]
    
    if 'global_state' in hyp:
        env.global_state = hyp['global_state']
    env_args = env.get_args(hyp['if_mac'])
    args = Arguments(env_args,hyp=hyp)

    log = args.init_before_training()

    memory = RandomMemory(args.max_capacity, args.cwd, args.if_load and not args.clear_memory, args.on_policy)
    if args.processes == 1:
        ctrl = args.agent_class(args.observ_space,args.action_shape,args)

    events = pd.read_csv(args.rainfall['training_events'])

    train_event_dir = os.path.splitext(args.swmm_input)[0] + '_train.inp'
    train_events = generate_split_file(args.swmm_input,filedir=train_event_dir,event_file=events[:args.explore_events],rain_num=args.explore_events,rain_arg=args.rainfall)

    eval_event_dir = os.path.splitext(args.swmm_input)[0] + '_eval.inp'
    eval_events = generate_split_file(args.swmm_input,filedir=eval_event_dir,event_file=events[-args.eval_events:],rain_num=args.eval_events,rain_arg=args.rainfall)

    # BC tests
    if args.processes > 1:
       pool = mp.Pool(args.processes)
       res = [pool.apply_async(func=bc_test,args=(env,event,)) for event in train_events+eval_events]
       pool.close()
       pool.join()
       bc_trains = [r.get() for r in res[:len(train_events)]]
       bc_evals = [r.get() for r in res[len(train_events):]]
    else:
       bc_trains = [bc_test(env,event) for event in train_events]
       bc_evals = [bc_test(env,event) for event in eval_events]

    # efd_trains = [efd_test(env,event) for event in train_events]
    # efd_evals = [efd_test(env,event) for event in eval_events]

    while args.episode <= args.total_episodes + args.pre_episodes:
        # Sampling
        if args.processes > 1:
            pool = mp.Pool(args.processes)
            res = []
            for event in train_events:
                r = pool.apply_async(func=interact_steps,args=(env,args,event,True,args.on_policy,))
                res.append(r)
            pool.close()
            pool.join()
            res = [r.get() for r in res]
        else:
            res = [interact_steps(env,ctrl,event,train=True,on_policy=args.on_policy)
             for event in train_events]
        trajs,rewards,perfs = [[r[i] for r in res] for i in range(3)]
        trajs = reduce(lambda x,y:x+y, trajs)
        memory.update(trajs)
        print('Sampling Complete: %s'%args.episode)

        if args.episode < args.pre_episodes:
            if args.processes > 1:
                args.episode_update()
            else:
                ctrl.episode_update(*args.episode_update())
            continue

        # Training
        print('Upgrading')
        if args.processes > 1:
            # memory,log = update_agent(args,memory,log,res)
            pool = mp.Pool(1)
            r = pool.apply_async(func=update_agent,args=(args,memory,log,res,))
            pool.close()
            pool.join()
            log = r.get()
            args.if_load = True
        else:
            losses = ctrl.update_net(memory)
            rewards,perfs = [r[1] for r in res],[r[-1] for r in res]
            update = log.log((rewards,perfs,losses),train=True)
            if update[0]:
                ctrl.save(os.path.join(ctrl.model_dir,'train'))
            if update[1]:
                ctrl.save(os.path.join(ctrl.model_dir,'reward'))
            # load & save in each episode after pre_episodes
        print('Upgrade Complete: %s'%args.episode)

        # on-policy
        if args.clear_memory:
            memory.clear()

        # Evaluate the model in several episodes
        if args.episode % args.eval_gap == 0:
            if args.processes > 1:
                pool = mp.Pool(args.processes)
                res = []
                for idx,event in enumerate(eval_events):
                    r = pool.apply_async(func=interact_steps,args=(env,args,event,False,args.on_policy,))
                    res.append(r)
                pool.close()
                pool.join()
                res = [r.get() for r in res]

                pool = mp.Pool(1)
                r = pool.apply_async(func=evaluate_agent,args=(args,log,res,))
                pool.close()
                pool.join()
                log = r.get()
            else:
                res = [interact_steps(env,ctrl,event,train=False,on_policy=args.on_policy)
                for event in enumerate(eval_events)]
                perfs = [r[-1] for r in res]
                losses = [ctrl.evaluate_net(r[0]) for r in res]
                update = log.log((perfs,losses),train=False)
                if update:
                    ctrl.save(os.path.join(ctrl.model_dir,'eval'))

        # Save the current model
        if args.episode % args.save_gap == 0:
            if args.processes == 1:
                ctrl.save()
            if not args.clear_memory:
                memory.save()
            log.save()
            # log.plot()

        # Update the episode and exploration greedy
        if args.processes == 1:
            ctrl.episode_update(*args.episode_update())
        else:
            args.episode_update()
            
    if args.processes == 1:
        ctrl.save()
    if not args.clear_memory:
        memory.save()
    log.save()
    log.plot()
