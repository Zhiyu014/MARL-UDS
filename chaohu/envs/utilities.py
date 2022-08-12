from math import log10
from numpy import diff, array
from os.path import exists,splitext
import random
from datetime import datetime,timedelta
from swmm_api import read_inp_file,read_rpt_file
from swmm_api.input_file.sections import Timeseries
from swmm_api.input_file.sections.others import TimeseriesData
import yaml

# This hyetograph is correct in a continous function, but incorrect with 5-min block.
def Chicago_Hyetographs(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    ts = []
    for i in range(dura//delta):
        t = i*delta
        key = str(1+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        if t <= r*dura:
            ts.append([key, (a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60])
        else:
            ts.append([key, (a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60])
    # tsd = TimeseriesData(Name = name,data = ts)
    return ts

# Generate a rainfall intensity file from a cumulative values in ICM
def Chicago_icm(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    HT = a*dura/(dura+b)**n
    Hs = []
    for i in range(dura//delta+1):
        t = i*delta
        if t <= r*dura:
            H = HT*(r-(r-t/dura)*(1-t/(r*(dura+b)))**(-n))
        else:
            H = HT*(r+(t/dura-r)*(1+(t-dura)/((1-r)*(dura+b)))**(-n))
        Hs.append(H)
    tsd = diff(array(Hs))*12
    ts = []
    for i in range(dura//delta):
        t = i*delta
        key = str(1+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        ts.append([key,tsd[i]])
    return ts


def generate_file(base_inp_file, para_file, pattern = 'Chicago_icm', filedir = None, rain_num = 1, replace = False):
    """
    Generate multiple inp files containing rainfall events
    designed by rainfall pattern.
    
    Parameters
    ----------
    base_inp_file : dir
        Path of the inp model.
    para_file : YAML
        Path of the rainfall pattern params.
    pattern : str, optional
        'Chicago_icm' or 'Chicago_Hyetographs'. The default is 'Chicago_icm'.
    filedir : dir, optional
        The output dir. The default is None.
    rain_num : int, optional
        numbers of rainfall events. The default is 1.

    Returns
    -------
    files : TYPE
        DESCRIPTION.

    """
    inp = read_inp_file(base_inp_file)
    paras = yaml.load(open(para_file, "r"), yaml.FullLoader)
    files = list()
    filedir = splitext(base_inp_file)[0] if filedir is None else filedir
    filedir = splitext(filedir)[0]+'_%s.inp'
    for i in range(rain_num):
        file = filedir%i
        files.append(file)
        if exists(file) == True:
            if replace:
                pass
            else:
                continue            
        p = random.randint(*paras['P'])
        delta = paras['delta']
        dura = paras['dura']
        simu_dura = paras['simu_dura']
        para = [random.uniform(*v)
                for k,v in paras['params'].items()] + [p,delta,dura]
        ts = eval(pattern)(para)

        inp['TIMESERIES'] = Timeseries.create_section()
        inp['TIMESERIES'].add_obj(TimeseriesData(Name = str(p)+'y',data = ts))
        inp.RAINGAGES['RG']['Timeseries'] = str(p)+'y'
        inp.RAINGAGES['RG']['Interval'] = str(int(delta//60)).zfill(2)+':'+str(int(delta%60)).zfill(2)
        start_time = datetime(2000,1,1,0,0)
        end_time = start_time + timedelta(minutes = simu_dura)
        inp.OPTIONS['START_DATE'] = start_time.date()
        inp.OPTIONS['END_DATE'] = end_time.date()
        inp.OPTIONS['START_TIME'] = start_time.time()
        inp.OPTIONS['END_TIME'] = end_time.time()
        inp.OPTIONS['REPORT_START_DATE'] = start_time.date()
        inp.OPTIONS['REPORT_START_TIME'] = start_time.time()
        inp.write_file(file)
    return files

def eval_control(event):
    if event.endswith('.inp'):
        event = event.replace('.inp','.rpt')
    rpt = read_rpt_file(event)
    flooding = rpt.flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
    outload = rpt.outfall_loading_summary['Total_Volume_10^6 ltr']
    cso = outload.sum() - outload.loc['WSC','Total_Volume_10^6 ltr']
    return 2*flooding + cso


def eval_closing_reward(event):
    '''
    Get the reward step-wise reward value
    Reward shaping:
    1. Operational reward: ±2
        1) If raining:  sum[(depth - ini_depth)/ini_depth] - sum[energy] * 0.2
        2) Not raining: sum[(ini_depth - depth)/ini_depth] - sum[energy] * 0.2
    2. Closing reward:  ±5 (gamma=0.95 n_steps=36 reward=2/(0.95**36)=4.16 --> 5)
        1) - 8 * (sum[flooding] + sum[overflow])/(sum[outflow] + sum[flooding] + final_storage) + 5
            cannot reach ±5 at all times so multiply -20 plus 15
        2) 10 * (HC[fl&CSO] - fl&CSO)/HC[fl&SCO]
            cannot reach ±5 at all times so multiply 10
    '''
    if event.endswith('.inp'):
        event = event.replace('.inp','.rpt')
    rpt = read_rpt_file(event)
    routing = rpt.flow_routing_continuity
    outload = rpt.outfall_loading_summary
    A,B = -20,10
    return A*(routing['Flooding Loss']['Volume_10^6 ltr'] +\
         routing['External Outflow']['Volume_10^6 ltr'] -\
             outload.loc['WSC','Total_Volume_10^6 ltr'] )/\
                sum([routing[col]['Volume_10^6 ltr']
                 for col in routing.keys() if 'Inflow' in col or 'Initial' in col]) + B
