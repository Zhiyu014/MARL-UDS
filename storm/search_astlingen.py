from envs.astlingen import astlingen
from envs.utilities import generate_split_file,get_depth_setting,get_flood_cso
from swmm_api import read_inp_file,read_rpt_file
import os
import multiprocessing as mp
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
import yaml
import pandas as pd
from utils.config import Arguments
HERE = os.path.dirname(__file__)

class mpc_problem(Problem):
    def __init__(self,eval_file,config):
        self.config = config
        self.file = eval_file
        self.n_pump = len(config.action_space)
        self.actions = list(config.action_space.values())
        self.n_step = config.control_horizon//config.control_interval
        self.n_var = self.n_pump*self.n_step
        self.n_obj = 1
            
        super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                         xl = np.array([0 for _ in range(self.n_var)]),
                         xu = np.array([len(v)-1 for _ in range(self.n_step)
                            for v in self.actions]),
                         vtype=int)

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_pump)).tolist()
        # eval_file = update_controls(self.file,self.config.action_space,k,y)
        # rpt_file,_ = swmm5_run(eval_file,create_out=False)

        env = astlingen(swmm_file = self.file)
        done = False
        reward = 0.0
        idx = 0
        while not done:
            done = env.step([self.actions[i][act] for i,act in enumerate(y[idx])])
            reward -= env.reward(norm=False)
            idx += 1
        return reward
        
    def _evaluate(self,x,out,*args,**kwargs):        
        pool = mp.Pool(self.config.processes)
        res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
        pool.close()
        pool.join()
        F = [r.get() for r in res]
        out['F'] = np.array(F)


def interact_steps(env,arg,event=None,train=True):
    if type(arg) is Arguments:
        f = arg.agent_class(arg.observ_space,arg.action_shape,arg,act_only=True)
    else:
        f = arg
    actions = []
    state = env.reset(event)
    done = False
    rewards = 0
    while not done:
        action = f.act(state,train)
        setting = f.convert_action_to_setting(action)
        done = env.step(setting,env.config['control_interval']*60)
        state = env.state()
        reward = env.reward(norm = True)
        rewards += reward
        actions.append(action)
    perf = env.performance('cumulative')
    return actions,perf

def initialize(x0,xl,xu,pop_size,prob):
    x0 = np.reshape(x0,-1)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [np.random.randint(xl[idx],xu[idx]+1) if np.random.random()<prob else x for idx,x in enumerate(x0)]
        population.append(xi)
    return np.array(population)

def simulate(file,controls):
    env = astlingen(swmm_file = file)
    done = False
    idx = 0
    while not done:
        done = env.step(controls[idx])
        idx += 1
    return env.performance('cumulative')


if __name__ == "__main__":
    
    env = astlingen()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']
    hyp_test.update(hyps['MPC'])
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()
    # logger.load(os.path.join(args.cwd,'MaxRed_all.json'))

    hyp = hyps['DQN']
    arg = Arguments(env.get_args(if_mac=hyp['if_mac']),hyp)
    arg.init_before_testing('reward')

    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    test_events = generate_split_file(args.swmm_input,filedir=test_event_dir,rain_num=args.test_events,rain_arg=args.rainfall)
    # test_events = ['./envs/network/astlingen/astlingen_test_%s.inp'%event for event in ['02_28_2007_12','07_24_2008_15','02_13_2009_15','10_01_2009_14']]

    cso_items = {ID:'creek' if weight == 2 else 'river' for ID,attr,weight in env.config['reward'] if attr == 'cumflooding'}
    for idx,event in enumerate(test_events):
        if event in logger.records.keys():
            continue
        inp = read_inp_file(event)
        start_time = datetime(inp.OPTIONS['START_DATE'].year,inp.OPTIONS['START_DATE'].month,inp.OPTIONS['START_DATE'].day,inp.OPTIONS['START_TIME'].hour,inp.OPTIONS['START_TIME'].minute)
        end_time = datetime(inp.OPTIONS['END_DATE'].year,inp.OPTIONS['END_DATE'].month,inp.OPTIONS['END_DATE'].day,inp.OPTIONS['END_TIME'].hour,inp.OPTIONS['END_TIME'].minute)

        rain_name = start_time.strftime('%m/%d/%Y-%H')

        actions,perf = interact_steps(env,arg,event,train=False)


        args.control_horizon = int((end_time-start_time).total_seconds()//60)

        prob = mpc_problem(event,args)
        sampling = initialize(actions,prob.xl,prob.xu,args.pop_size,args.sampling[-1])
        # sampling = eval(args.sampling[0])()
        crossover = eval(args.crossover[0])(vtype=int,repair=RoundingRepair())
        mutation = eval(args.mutation[0])(*args.mutation[1:],vtype=int,repair=RoundingRepair())

        method = GA(pop_size = args.pop_size,
                    sampling = sampling,
                    crossover = crossover,
                    mutation = mutation,
                    eliminate_duplicates=True)
        
        res = minimize(prob,
                    method,
                    #    termination = args.termination,
                    # save_history=True,
                    verbose=True)
        print("Best solution found: %s" % res.X)
        print("Function value: %s" % res.F)
        controls = res.X.reshape((prob.n_step,prob.n_pump)).tolist()
        controls = [[prob.actions[i][act] for i,act in enumerate(control)]
         for control in controls]
        perf = simulate(event,controls)


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

        logger.log((target.to_json(),operat.to_json(),perf),name='MaxRed',event=rain_name)
        logger.save(os.path.join(logger.cwd,'MaxRed.json'))
