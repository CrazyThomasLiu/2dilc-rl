from run import *
from agent import AgentModSAC,AgentSAC
import pdb
import control
from Nonlinearsys_time_batch_v4 import NonlinearSysEnv
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
# create Agent
args = Arguments(if_on_policy=False)
args.agent =AgentModSAC()
#args.agent =AgentSAC()
#pdb.set_trace()
#args.agent.if_use_gae = True
args.break_step = 180*2000
#args.agent.lambda_entropy = 0.04
#args.agent.lambda_entropy = 0.08
#print(torch.get_default_dtype())
#pdb.set_trace()

#create Nonlinearsys Env
"""
def updatestep(t, x, u, params):
    # Parameter setup

    # Map the states into local variable names
    # a=0.5+0.10000*np.sin(0.1*t)
    # pdb.set_trace()
    # Compute the discrete updates
    # pdb.set_trace()
    dY = -np.sin(x) + (0.5 + 0.10000 * np.sin(0.1 * t)) * u

    return [dY]
"""
def updatestep(t, x, u, params):
    # Parameter setup
    batch_num = params.get('batch_num', 0)
    # Map the states into local variable names
    "time"
    a=0.5+0.1*np.sin(0.1*t)
    "batch"
    b=0.05*np.sin(batch_num * 2 * np.pi / 20)
    # pdb.set_trace()
    # Compute the discrete updates
    # pdb.set_trace()

    dY = -np.sin(x) + (a+b)* u

    return [dY]


sys = control.NonlinearIOSystem(
    updatestep, None, inputs=('u'), outputs=('x'),
    states=('x'), name='nonlinear')

N=180   # the duration of each batch

args.env = NonlinearSysEnv(N,sys)
args.env_eval = NonlinearSysEnv(N,sys)


args.env.target_return = 200000
args.env_eval.target_return = 200000
# Hyperparameters
args.agent.cri_target = True
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
#args.reward_scale = 2 ** -10000  # RewardRange: -1800 < -200 < -50 < 0
#args.gamma = 0.99
args.gamma = 0.99
"""
Jianan Liu
args.net_dim = 2 ** 5000
#args.net_dim = 2 ** 8
args.batch_size = args.net_dim * 2
#############################
args.net_dim = 2 ** 8
args.batch_size = 2 ** 8
args.target_step = args.env.max_step * 2
args.net_dim = 2 ** 6
args.batch_size = 2 ** 6
args.target_step = args.env.max_step * 2
"""
#args.net_dim = 2 ** 6
#args.net_dim = 2 ** 5000
args.net_dim = 2 ** 8
#args.net_dim = 2 ** 9
args.batch_size = args.net_dim * 2
#args.batch_size =  N
args.target_step = 3*N
#args.target_step = args.env.max_step * 8
#args.target_step=8
"""
author:jianan liu
"""
#args.max_memo = args.env.max_step*10
#pdb.set_trace()
args.repeat_times=1
args.soft_update_tau = 2 ** -8
#args.soft_update_tau = 2 ** -9
#args.soft_update_tau = 2 ** -10
#args.learning_rate=0.0001
#args.learning_rate=9e-5
args.learning_rate=3e-4
args.eval_gap=2 ** 6
args.eval_times1 = 20
args.eval_times2 = 60
#args.eval_gap=2 ** 2
#args.learning_rate=0.001
##############################################
#pprint.pprint(args.__dict__)
#pdb.set_trace()
train_and_evaluate(args)