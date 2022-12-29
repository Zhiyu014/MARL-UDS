from envs.chaohu import chaohu
from envs.utilities import generate_file,get_depth_setting,get_flood_cso,eval_control,eval_pump
from swmm_api import read_inp_file
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
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
        self.n_step = config.control_horizon//config.control_interval
        self.n_var = self.n_pump*self.n_step
        self.n_obj = 1
            
        super().__init__(n_var=self.n_var, n_obj=self.n_obj,
                        n_ieq_constr=3*self.n_step,
                        #  xl = np.array([0 for _ in range(self.n_var)]),
                        #  xu = np.array([len(v)-1 for _ in range(self.n_step)
                        #     for v in config.action_space.values()]),
                         vtype=bool)

    def pred_simu(self,y):
        y = y.astype(int).reshape((self.n_step,self.n_pump)).tolist()
        # eval_file = update_controls(self.file,self.config.action_space,k,y)
        # rpt_file,_ = swmm5_run(eval_file,create_out=False)

        env = chaohu(swmm_file = self.file)
        for ID,attr,_ in env.config['performance_targets']:
            env.data_log[attr][ID].append(
                env.env.methods[attr](ID)
            )
        done = False
        # reward = 0
        idx = 0
        while not done:
            done = env.step(y[idx])
            # reward -= env.reward()
            idx += 1
        flood = env.data_log['cumflooding']['system']
        perf = 2* (flood[-1]-flood[0]) + sum([env.data_log['totalinflow'][idx][-1]-env.data_log['totalinflow'][idx][0]
         for idx,attr,_ in env.config['performance_targets'] if attr == 'totalinflow'])
        return perf
        
    def _evaluate(self,x,out,*args,**kwargs):

        # out["F"] = zeros((x.shape[0],self.n_obj))
        
        pool = mp.Pool(self.config.processes)
        res = [pool.apply_async(func=self.pred_simu,args=(xi,)) for xi in x]
        pool.close()
        pool.join()
        F = [r.get() for r in res]
        out['F'] = np.array(F)

        # pool = ThreadPool(self.config.threads)
        # F = pool.starmap(self.para_eval,params)
        # out['F'] = np.array(F)

        # TODO: How to encode constraints in yaml?
        consts = []
        y = x.astype(int)
        for idx in range(self.n_step):
            consts += [y[:,self.n_pump*idx+1]-y[:,self.n_pump*idx], y[:,self.n_pump*idx+3]-y[:,self.n_pump*idx+2], y[:,self.n_pump*idx+6]-y[:,self.n_pump*idx+5]]
        out['G'] = np.column_stack(consts)

def initialize(x0,pop_size,prob):
    x0 = np.reshape(x0,-1)
    population = [x0]
    for _ in range(pop_size-1):
        xi = [bool(np.random.randint(0,2)) if np.random.random()<prob else bool(x) for x in x0]
        population.append(xi)
    return np.array(population)


def run_ea(eval_file,args,setting=None):
    prob = mpc_problem(eval_file,args)
    if setting is not None:
        sampling = initialize(setting,args.pop_size,args.sampling[-1])
    else:
        sampling = eval(args.sampling[0])()
    crossover = eval(args.crossover[0])(vtype=bool,repair=RoundingRepair())
    mutation = eval(args.mutation[0])(*args.mutation[1:],vtype=bool,repair=RoundingRepair())

    method = GA(pop_size = args.pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                eliminate_duplicates=True)
    
    res = minimize(prob,
                   method,
                   termination = args.termination,
                   seed = args.seed,
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
    ctrls = ctrls.reshape((prob.n_step,prob.n_pump)).astype(int).tolist()
    return ctrls[0]

    

def mpc_test(env,args,event=None,settings = None,arg=None):
    state = env.reset(event)
    if arg is not None:
        f = arg.agent_class(arg.observ_space,arg.action_shape,arg)
    done = False
    idx = 0
    while not done:
        # get initial setting
        # cur_setting = [env.data_log['setting'][ID][-1]
        # for ID in env.config['action_space'] if len(env.data_log['setting'][ID])>0]
        # cur_setting = [0 for _ in env.config['action_space']] if cur_setting == [] else cur_setting
        # settings = cur_setting * (arg.control_horizon//arg.control_interval)

        setting = settings[idx:idx+args.control_horizon//args.control_interval] if settings is not None else None

        # get agent reactions
        if arg is not None:
            action = f.act(state,False)
            setting = f.convert_action_to_setting(action)

        # get eval and hsf file
        eval_file = env.get_eval_file()

        # run predictive optimization
        setting = run_ea(eval_file,args,setting)
        print("Finish search: %s"%env.env.methods['simulation_time']())

        done = env.step(setting)
        state = env.state()

    perf = env.performance('cumulative')
    return perf

if __name__ == '__main__':
    # init SWMM environment and arguments
    env = chaohu()
    hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)[env.config['env_name']]
    hyp_test = hyps['test']
    hyp_test.update(hyps['MPC'])

    # init test args
    args = Arguments(env.get_args(), hyp_test)
    logger = args.init_test()
    # logger.load(os.path.join(args.cwd,'%s.json'%args.test_name))
    
    # init reactive RL agent
    # hyp = hyps['DQN']
    # if hyp_test['if_predict']:
    #     hyp.update(hyps['predict'])
    # env_args = env.get_args(if_mac=hyp['if_mac'])
    # arg = Arguments(env_args,hyp)
    # arg.init_before_testing('reward')

    # generate rainfall
    test_event_dir = os.path.splitext(args.swmm_input)[0] + '_test.inp'
    rainpara = yaml.load(open(args.rainfall_parameters, "r"), yaml.FullLoader)
    rainpara['P'] = [1,2,3,5]
    rainpara['params'].update({'A':25.828,'C':1.3659,'n':0.9126,'b':20.515,'r':0.375})
    test_events = generate_file(args.swmm_input,
                                rainpara,
                                # args.rainfall_parameters,
                                filedir=test_event_dir,
                                rain_num=4,
                                replace=False)


    for idx,event in enumerate(test_events):
        rain_name = 'Rain %s'%(idx+1)
        P = read_inp_file(event).RAINGAGES['RG']['Timeseries']

        # Use maxred settings as initial values of MPC
        # max_logger = Testlogger(args.cwd)
        # max_logger.load(os.path.join(args.cwd,'mared.json'))
        # operat = pd.read_json(logger.records[rain_name]['operation']['MaxRed'])
        # settings = operat[list(args.action_space)].to_numpy().tolist()
        # settings = [sett for idx,sett in enumerate(settings) if idx % args.control_interval == args.control_interval-1]

        perf = mpc_test(env,args,event,arg=None)
        print('MPC Score at event {0}: {1}'.format(idx,perf))

        target = get_flood_cso(event,args.outfall,cumulative=True)
        operat = get_depth_setting(event,args.storage,list(env.config['action_space'].keys()))
        
        flooding,cso = eval_control(event)
        energy = eval_pump(event,list(env.config['action_space'].keys()))
        perf = {'System flooding':flooding,'CSO':cso,'Pumping energy':energy}

        logger.log((target.to_json(),operat.to_json(),perf),name='MPC',event=rain_name,P = P)

        logger.save(os.path.join(logger.cwd,'%s.json'%args.test_name))
