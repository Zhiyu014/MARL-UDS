
from envs.chaohu import chaohu
from envs.utilities import generate_file,get_depth_setting,get_flood_cso
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
import yaml
import os
import multiprocessing as mp
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)

# TODO add predictive control codes
def predict(event,args,f='VDN'):
    env = chaohu(swmm_file = event)
    ctrl = args.agent_class(args.observ_space,args.action_space,args)
    done = False
    actions = []
    while not done:
        state = env.state()
        action = ctrl.act(state,train=False)
        actions.append(action)
        setting = ctrl.convert_action_to_setting(action)
        done = env.step(setting)
        # perf = env.performance()
    return actions

def interact_steps(env,f,event=None,train=True,if_predict=False):
    state = env.reset(event)
    done = False
    while not done:
        # no prediction
        action = f.act(state,train)
        setting = f.convert_action_to_setting(action)
        # TODO prediction
        # if if_predict:
        #     eval_file = env.get_eval_file()
        #     q = mp.Queue()
        #     p = mp.Process(target=predict,args=(eval_file,args,))
        #     p.start()
        #     p.join()
        #     actions = q.get()
        #     ctrl = run_ea(eval_file,actions,args)
        done = env.step(setting)
        state = env.state()
    perf = env.performance('cumulative')
    return perf


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
    # ctrls = args.init_ctrls
    ctrls = {}
    for agent in hyp_test['test_agents']:
        hyp = hyps[agent]
        env_args = env.get_args(if_mac=hyp['if_mac'])
        args = Arguments(env_args,hyp)
        args.init_before_testing()
        ctrls[agent] = args.agent_class(args.observ_space,args.action_space,args)

    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()

    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'

    test_events = generate_file(args.swmm_input,args.rainfall_parameters,
                                 filedir=test_event_dir,
                                 rain_num=args.test_events,
                                 replace=args.replace_rain)

    # test in rainfall events
    for idx,event in enumerate(test_events):
        rain_name = 'Rain %s'%idx
        perf = hc_test(env,event)
        print('HC Score at event {0}: {1}'.format(idx,perf))

        target = get_flood_cso(event,args.outfall,cumulative=True)
        operat = get_depth_setting(event,args.storage,env.config['action_space'])
        logger.log((target,operat),agent='HC',event=rain_name)


        for agent,ctrl in ctrls.items():
            perf = interact_steps(env,ctrl,event,train=False)
            print('{0} Testing Score at event {1}: {2}'.format(agent,idx,perf))
            
            target = get_flood_cso(event,args.outfall,cumulative=True)
            operat = get_depth_setting(event,args.storage,env.config['action_space'])
            logger.log((target,operat),agent)

    logger.save()