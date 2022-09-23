from swmm_api import read_inp_file
from envs.chaohu import chaohu
from envs.utilities import generate_file,get_depth_setting,get_flood_cso,eval_control,eval_pump
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
import yaml
import os
import multiprocessing as mp
from ea import run_ea
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

# TODO add predictive control codes
def predict(event,arg,q):
    env = chaohu(swmm_file = event)
    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    done = False
    settings = []
    while not done:
        state = env.state()
        action = f.act(state,train=False)
        setting = f.convert_action_to_setting(action)
        settings.append(setting)
        done = env.step(setting)
    q.put(settings)

def interact_steps(env,arg,event=None,train=False,if_predict=False):
    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    state = env.reset(event)
    done = False
    while not done:
        if if_predict:
            # get current setting
            cur_setting = [env.data_log['setting'][ID][-1]
            for ID in env.config['action_space'] if len(env.data_log['setting'][ID])>0]
            cur_setting = [0 for _ in env.config['action_space']] if cur_setting == [] else cur_setting
            
            # get agent reactions
            eval_file = env.get_eval_file()
            q = mp.Queue()
            p = mp.Process(target=predict,args=(eval_file,arg,q,))
            p.start()
            p.join()
            print("Finish reaction: %s"%env.env.methods['simulation_time']())
            
            # run predictive optimization
            settings = [cur_setting] + q.get()
            setting = run_ea(eval_file,settings,arg)
            print("Finish search: %s"%env.env.methods['simulation_time']())

        else:
            # no prediction
            action = f.act(state,train)
            setting = f.convert_action_to_setting(action)
        done = env.step(setting)
        state = env.state()
    perf = env.performance('cumulative')
    return perf

def interact_steps_fail(env,arg,event=None,observ_fail=False,act_fail=False,backup=None):
    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    if act_fail and type(backup) == type(arg):
        bk = backup.agent_class(backup.observ_space,backup.action_shape,backup)
    state = env.reset(event)
    done = False
    while not done:
        if observ_fail:
            state = state[0] + [s*int(random.random()>observ_fail) for s in state[1:]]       
        action = f.act(state,False)
        if act_fail:
            fail = {k:int(random.random()>act_fail) for k in arg.site}
            fail = [fail[act[:2]] for act in arg.action_space]
            if backup is None:
                setting = [sett*fail[idx] + setting[idx] * (1-fail[idx])
                for idx,sett in enumerate(f.convert_action_to_setting(action))]
            elif backup is hc_controller:
                hc_setting = hc_controller(env.env._state()[1:3],setting)
                setting = [sett*fail[idx] + hc_setting[idx] * (1-fail[idx])
                for idx,sett in enumerate(f.convert_action_to_setting(action))]
            elif type(backup) == type(arg):
                bk_setting = bk.act(state,False)
                setting = [sett*fail[idx] + bk_setting[idx] * (1-fail[idx])
                for idx,sett in enumerate(f.convert_action_to_setting(action))]

        else:
            setting = f.convert_action_to_setting(action)

        done = env.step(setting)
        state = env.state()
    perf = env.performance('cumulative')
    return perf

def hc_controller(depth,setting):
    starts = [int(depth[0]>h) for h in [0.8,1,1.2,1.4]] + [int(depth[1]>h) for h in [4,4.2,4.3]]
    shuts = [1-int(depth[0]<0.5) for _ in range(4)] + [1-int(depth[1]<h) for h in [1,1,2,1.2]]
    setting = [max(sett,starts[i]) for i,sett in enumerate(setting)]
    setting = [min(sett,shuts[i]) for i,sett in enumerate(setting)]
    return setting

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
    env = chaohu()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']

    # init control agents
    ctrls = {}
    for agent in hyp_test['test_agents']:
        hyp = hyps[agent]
        # update search parameters in the agent arg
        if hyp_test['if_predict']:
            hyp.update(hyps['predict'])
        env_args = env.get_args(if_mac=hyp['if_mac'])
        arg = Arguments(env_args,hyp)
        arg.init_before_testing()
        ctrls[agent] = arg

    # init test args
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()

    # generate rainfall
    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    rainpara = yaml.load(open(args.rainfall_parameters, "r"), yaml.FullLoader)
    rainpara['P'] = [1,2,3,5]
    rainpara['params'].update({'A':25.828,'C':1.3659,'n':0.9126,'b':20.515,'r':0.375})
    test_events = generate_file(args.swmm_input,
                                rainpara,
                                # args.rainfall_parameters,
                                filedir=test_event_dir,
                                rain_num=args.test_events,
                                replace=args.replace_rain)

    # test in rainfall events
    for idx,event in enumerate(test_events):
        rain_name = 'Rain %s'%(idx+1)
        P = read_inp_file(event).RAINGAGES['RG']['Timeseries']
        perf = hc_test(env,event)
        print('HC Score at event {0}: {1}'.format(idx,perf))

        target = get_flood_cso(event,args.outfall,cumulative=True)
        operat = get_depth_setting(event,args.storage,list(env.config['action_space'].keys()))
        
        flooding,cso = eval_control(event)
        energy = eval_pump(event,list(env.config['action_space'].keys()))
        perf = {'System flooding':flooding,'CSO':cso,'Pumping energy':energy}

        logger.log((target,operat,perf),name='HC',event=rain_name,P = P)


        for agent,arg in ctrls.items():
            perf = interact_steps(env,arg,event,train=False,if_predict=args.if_predict)
            print('{0} Testing Score at event {1}: {2}'.format(agent,idx,perf))
            
            target = get_flood_cso(event,args.outfall,cumulative=True)
            operat = get_depth_setting(event,args.storage,list(env.config['action_space'].keys()))
            
            flooding,cso = eval_control(event)
            energy = eval_pump(event,list(env.config['action_space'].keys()))
            perf = {'System flooding':flooding,'CSO':cso,'Pumping energy':energy}

            name = agent + '_predict' if args.if_predict else agent
            logger.log((target,operat,perf),name)

    logger.save(os.path.join(logger.cwd,'records.json'))
    # logger.save()