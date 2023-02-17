from swmm_api import read_inp_file
from envs.astlingen import astlingen
from envs.utilities import generate_split_file
from envs.utilities import get_depth_setting,get_flood_cso
from swmm_api import read_rpt_file
# from rnmemory import Recurrent_RandomMemory
from utils.config import Arguments
import yaml
import os
import pandas as pd
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = os.path.dirname(__file__)


def interact_steps(env,arg,event=None,train=False,on_policy=False):
    f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    state = env.reset(event)
    done = False
    while not done:
        # no prediction
        action = f.act(state,train)
        if on_policy:
            action,_ = action
        setting = f.convert_action_to_setting(action)
        done = env.step(setting)
        state = env.state()
    perf = env.performance('cumulative')
    return perf



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

    # init test args
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()

    # generate rainfall
    # events = pd.read_csv(args.rainfall['rainfall_events'].replace('rain','train'))
    # train_event_dir = os.path.splitext(args.swmm_input)[0] + '_train.inp'
    # train_events = generate_split_file(args.swmm_input,filedir=train_event_dir,event_file=events[:50],rain_num=50,rain_arg=args.rainfall)
    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    test_events = generate_split_file(args.swmm_input,filedir=test_event_dir,rain_num=args.rainfall['test_events'],rain_arg=args.rainfall)


    cso_items = {ID:'creek' if weight == 2 else 'river' for ID,attr,weight in env.config['reward'] if attr == 'cumflooding'}
    for event in test_events:
        rain_name = read_inp_file(event).OPTIONS['START_DATE'].strftime('%m/%d/%Y') + '-' + read_inp_file(event).OPTIONS['START_TIME'].strftime('%H')

        for agent,arg in list(ctrls.items()) + [('BC',None)]:
            if agent == 'BC':
                perf = bc_test(env,event)
                print('BC Score at event {0}: {1}'.format(rain_name,perf))
            else:
                perf = interact_steps(env,arg,event,train=False)
                print('{0} Testing Score at event {1}: {2}'.format(agent,rain_name,perf))

            target = get_flood_cso(event,cumulative=False)
            operat = get_depth_setting(event,[ID for ID,_ in env.config['states'] if ID.startswith('T')],list(env.config['action_space']))
            
            inp = read_inp_file(event)
            for k,v in inp.TIMESERIES.items():
                rain = pd.DataFrame(v.data,columns=['time',k]).set_index('time')
                target[k] = rain
                operat[k] = rain
            target = target.fillna(0.0)
            operat = operat.fillna(0.0)

            node_flooding = read_rpt_file(event.replace('.inp','.rpt')).node_flooding_summary
            perf = {'creek':0,'river':0,'cso':0}
            if node_flooding is not None:
                flood_volume = node_flooding['Total_Flood_Volume_10^6 ltr']
                for k,v in flood_volume.to_dict().items():
                    perf[cso_items[k]] += v
                perf['cso'] = flood_volume.sum()
                
            logger.log((target.to_json(),operat.to_json(),perf),name=agent,event=rain_name)

    logger.save(os.path.join(logger.cwd,'%s.json'%args.test_name))
