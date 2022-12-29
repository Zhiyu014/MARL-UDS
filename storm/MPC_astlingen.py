from envs.astlingen import astlingen
from envs.utilities import generate_split_file,get_depth_setting,get_flood_cso,eval_control,eval_pump
from swmm_api import read_inp_file,read_rpt_file
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
import yaml
from utils.config import Arguments
# from utils.logger import Testlogger
HERE = os.path.dirname(__file__)

class mpc_problem(Problem):
    def __init__(self,eval_file,config):
        self.config = config
        self.file = eval_file
        self.n_pump = len(config.action_space)
        self.actions = list(config.action_space.values())
        self.n_step = config.prediction['control_horizon']//config.control_interval
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
        idx = 0
        reward = 0
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

def initialize(x0,xl,xu,pop_size,prob):
    x0 = np.reshape(x0,-1)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [np.random.randint(xl[idx],xu[idx]+1) if np.random.random()<prob else x for idx,x in enumerate(x0)]
        population.append(xi)
    return np.array(population)


def run_ea(eval_file,args,setting=None):
    prob = mpc_problem(eval_file,args)
    if setting is not None:
        sampling = initialize(setting,prob.xl,prob.xu,args.pop_size,args.sampling[-1])
    else:
        sampling = eval(args.sampling[0])()
    crossover = eval(args.crossover[0])(vtype=int,repair=RoundingRepair())
    mutation = eval(args.mutation[0])(*args.mutation[1:],vtype=int,repair=RoundingRepair())

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    
    res = minimize(prob,
                   method,
                   termination = args.termination,
                   verbose = True)
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)

    # Multiple solutions with the same performance
    # Choose the minimum changing
    # if res.X.ndim == 2:
    #     X = res.X[:,:prob.n_pump]
    #     chan = (X-np.array(settings)).sum(axis=1)
    #     ctrls = res.X[chan.argmin()]
    # else:
    ctrls = res.X
    ctrls = ctrls.reshape((prob.n_step,prob.n_pump)).tolist()
    return ctrls[0]

    

def mpc_test(env,arg,event=None,settings = None):
    _ = env.reset(event)
    done = False
    idx = 0
    while not done:
        # get initial setting
        # cur_setting = [env.data_log['setting'][ID][-1]
        # for ID in env.config['action_space'] if len(env.data_log['setting'][ID])>0]
        # cur_setting = [0 for _ in env.config['action_space']] if cur_setting == [] else cur_setting
        # settings = cur_setting * (arg.control_horizon//arg.control_interval)
        # get agent reactions
        eval_file = env.get_eval_file()
        
        # run predictive optimization
        
        setting = settings[idx:idx+arg.prediction['control_horizon']//arg.control_interval] if settings is not None else None

        setting = run_ea(eval_file,arg,setting)
        print("Finish search: %s"%env.env.methods['simulation_time']())

        done = env.step(setting)
    perf = env.performance('cumulative')
    return perf

if __name__ == '__main__':
    # init SWMM environment and arguments
    env = astlingen()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']
    hyp_test.update(hyps['MPC'])

    # init test args
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()
    # logger.load(os.path.join(args.cwd,'%s.json'%args.test_name))

    # generate rainfall
    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    test_events = generate_split_file(args.swmm_input,filedir=test_event_dir,rain_num=args.test_events,rain_arg=args.rainfall)


    cso_items = {ID:'creek' if weight == 2 else 'river' for ID,attr,weight in env.config['reward'] if attr == 'cumflooding'}
    for idx,event in enumerate(test_events):
        inp = read_inp_file(event)
        start_time = datetime(inp.OPTIONS['START_DATE'].year,inp.OPTIONS['START_DATE'].month,inp.OPTIONS['START_DATE'].day,inp.OPTIONS['START_TIME'].hour,inp.OPTIONS['START_TIME'].minute)
        rain_name = start_time.strftime('%m/%d/%Y-%H')

        # Use maxred settings as initial values of MPC
        # operat = pd.read_json(logger.records[rain_name]['operation']['MaxRed'])
        # settings = operat[list(args.action_space)].to_numpy().tolist()
        # settings = [sett for idx,sett in enumerate(settings) if idx % args.control_interval == args.control_interval-1]

        perf = mpc_test(env,args,event,settings=None)
        print('MPC Score at event {0}: {1}'.format(idx,perf))


        target = get_flood_cso(event,cumulative=True)
        operat = get_depth_setting(event,[ori.replace('V','T') for ori in env.config['action_space']],list(env.config['action_space']))
        
        node_flooding = read_rpt_file(event.replace('.inp','.rpt')).node_flooding_summary
        perf = {'creek':0,'river':0,'cso':0}
        if node_flooding is not None:
            flood_volume = node_flooding['Total_Flood_Volume_10^6 ltr']
            for k,v in flood_volume.to_dict().items():
                if k not in cso_items:
                    cso_items[k] = 'river'
                perf[cso_items[k]] += v
            perf['cso'] = flood_volume.sum()

        logger.log((target.to_json(),operat.to_json(),perf),name='MPC',event=rain_name)

        logger.save(os.path.join(logger.cwd,'%s.json'%args.test_name))
