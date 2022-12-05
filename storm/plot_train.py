from utils.logger import Trainlogger
import yaml,os,sys
HERE = os.path.dirname(__file__)

if __name__ == '__main__':
    env_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        cwd = sys.argv[2]
    else:
        hyps = yaml.load(open(os.path.join(HERE,'utils','config.yaml'), "r"), yaml.FullLoader)
        hyp = hyps[env_name]
        cwd = hyp[hyp['train']]['cwd']
    log = Trainlogger(cwd,True)
    log.plot()