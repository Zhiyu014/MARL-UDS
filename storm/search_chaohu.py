from envs.chaohu import chaohu
from envs.utilities import generate_file,get_depth_setting,get_flood_cso,eval_control,eval_pump
from swmm_api import read_inp_file
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import Problem
import yaml
from utils.config import Arguments
HERE = os.path.dirname(__file__)


def interact_steps(env,arg,event=None):
    if type(arg) is Arguments:
        f = arg.agent_class(arg.observ_space,arg.action_shape,arg,act_only=True)
    else:
        f = arg
    actions = []
    state = env.reset(event)
    done = False
    while not done:
        action = f.act(state,False)
        setting = f.convert_action_to_setting(action)
        done = env.step(setting,env.config['control_interval']*60)
        actions.append(action)    
    return actions

class mpc_problem(Problem):
    def __init__(self,eval_file,config):
        self.config = config
        self.file = eval_file
        self.n_pump = len(config.action_space)
        self.n_step = config.control_horizon//config.control_interval
        self.n_var = self.n_pump*self.n_step
        self.n_obj = 1
            
        super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                        # n_ieq_constr=3*self.n_step,
                        #  xl = np.array([0 for _ in range(self.n_var)]),
                        #  xu = np.array([len(v)-1 for _ in range(self.n_step)
                            # for v in config.action_space.values()]),
                         vtype=bool)

    def pred_simu(self,y):
        y = y.reshape((self.n_step,self.n_pump)).tolist()
        # eval_file = update_controls(self.file,self.config.action_space,k,y)
        # rpt_file,_ = swmm5_run(eval_file,create_out=False)

        env = chaohu(swmm_file = self.file)
        done = False
        # reward = 0
        idx = 0
        while not done:
            done = env.step(y[idx])
            # reward -= env.reward()
            idx += 1
        perf = 2* env.data_log['cumflooding']['system'][-1] + sum([env.data_log['totalinflow'][idx][-1] for idx,attr,_ in env.config['performance_targets'] if attr == 'totalinflow'])
        # perf = env.performance('cumulative')
        return perf
        
    def _evaluate(self,x,out,*args,**kwargs):        
        pool = mp.Pool(self.config.processes)
        res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
        pool.close()
        pool.join()
        F = [r.get() for r in res]
        out['F'] = np.array(F)/1e3

        # consts = []
        # y = x.astype(int)
        # for idx in range(self.n_step):
        #     consts += [y[:,self.n_pump*idx+1]-y[:,self.n_pump*idx], 
        #     y[:,self.n_pump*idx+3]-y[:,self.n_pump*idx+2], 
        #     y[:,self.n_pump*idx+6]-y[:,self.n_pump*idx+5]]
        # out['G'] = np.column_stack(consts)

def initialize(x0,pop_size,prob):
    x0 = np.reshape(x0,-1).astype(bool)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [bool(np.random.randint(0,2)) if np.random.random()<prob else bool(x) for x in x0]
        population.append(xi)
    return np.array(population)

def simulate(file,controls):
    env = chaohu(swmm_file = file)
    done = False
    idx = 0
    while not done:
        done = env.step(controls[idx])
        idx += 1
    return env.performance('cumulative')

def hc_test(env,event=None):
    def hc_controller(depth,setting):
        starts = [int(depth[0]>h) for h in [0.8,1,1.2,1.4]] + [int(depth[1]>h) for h in [4,4.2,4.3]]
        shuts = [1-int(depth[0]<0.5) for _ in range(4)] + [1-int(depth[1]<h) for h in [1,1.2,1.2]]
        setting = [max(sett,starts[i]) for i,sett in enumerate(setting)]
        setting = [min(sett,shuts[i]) for i,sett in enumerate(setting)]
        return setting

    _ = env.reset(event)
    actions = []
    setting = [0 for _ in env.config['action_space']]
    done = False
    while not done:
        depth = env.env._state()[1:3]
        setting = hc_controller(depth,setting)
        actions.append(setting)
        done = env.step(setting,env.config['control_interval']*60)
    perf = env.performance('cumulative')
    return perf,actions
    
if __name__ == "__main__":
    
    env = chaohu()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']
    hyp_test.update(hyps['MPC'])
    args = Arguments(env.get_args(), hyp_test)
    args.control_horizon = 300
    logger = args.init_test()
    logger.load('./results/chaohu/records4.json')
    # hyp = hyps['DQN']
    # arg = Arguments(env.get_args(if_mac=hyp['if_mac']),hyp)
    # arg.init_before_testing('reward')
    # ini_log = args.init_test()
    # ini_log.load('./results/chaohu_ori2/records4.json')

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

    
    for idx,event in enumerate(test_events):
        rain_name = 'Rain %s'%(idx+1)
        P = read_inp_file(event).RAINGAGES['RG']['Timeseries']


        prob = mpc_problem(event,args)
        # actions = interact_steps(env,arg,event)
        perf,actions = hc_test(env,event)
        print('HC perf: %s'%perf)

        actions = pd.read_json(logger.records[rain_name]['operation']['DQN'])[list(env.config['action_space'])].to_numpy().tolist()
        actions = [act for idx,act in enumerate(actions[:300]) if idx%10==0]
        
        sampling = initialize(actions,args.pop_size,args.sampling[-1])
        # sampling = eval(args.sampling[0])()
        crossover = eval(args.crossover[0])(vtype=bool,repair=RoundingRepair())
        mutation = eval(args.mutation[0])(*args.mutation[1:],vtype=bool,repair=RoundingRepair())

        method = GA(pop_size = args.pop_size,
                    sampling = sampling,
                    crossover = crossover,
                    mutation = mutation,
                    eliminate_duplicates=True)
        
        res = minimize(prob,
                    method,
                    #    termination = args.termination,
                    # seed = args.seed,
                    # save_history=True,
                    verbose=True)
        print("Best solution found: %s" % res.X)
        print("Function value: %s" % res.F)

        controls = res.X.reshape((prob.n_step,prob.n_pump)).astype(int).tolist()
        perf = simulate(event,controls)

        target = get_flood_cso(event,args.outfall,cumulative=True)
        operat = get_depth_setting(event,args.storage,list(env.config['action_space'].keys()))
        
        flooding,cso = eval_control(event)
        energy = eval_pump(event,list(env.config['action_space'].keys()))
        perf = {'System flooding':flooding,'CSO':cso,'Pumping energy':energy}

        logger.log((target.to_json(),operat.to_json(),perf),name='MaxRed',event=rain_name,P = P)
        logger.save(os.path.join(logger.cwd,'records4.json'))

