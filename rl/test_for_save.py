from run_for_implementation import *
from agent import AgentModSAC,AgentSAC
import pdb
import control
#from cstr_MIMO.env_MIMO_cstr_40_time_batch_constrain150 import BatchSysEnv
#from cstr_MIMO.env_MIMO_cstr_30_time_batch_constrain1000_n1000 import BatchSysEnv
#from cstr_MIMO.env_MIMO_cstr_30_time_batch_constrain2000_n2000 import BatchSysEnv
#from cstr_MIMO.env_MIMO_cstr_30_time_batch_constrain20_2000 import BatchSysEnv
#from cstr_MIMO.env_MIMO_cstr_30_time_batch_constrain5000_n5000 import BatchSysEnv
#from cstr_MIMO.env_MIMO_cstr_30_constrain5000_n5000_NonconstrainforILC import BatchSysEnv
from env_linear_injection_modeling import BatchSysEnv
import numpy as np
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
import copy
import os
# create Agent
args = Arguments(if_on_policy=False)
args.agent =AgentModSAC()
#args.agent =AgentSAC()
#pdb.set_trace()
#args.agent.if_use_gae = True
args.break_step = 200*50
#args.agent.lambda_entropy = 0.04
#args.agent.lambda_entropy = 0.08
#print(torch.get_default_dtype())
#pdb.set_trace()

"""create batch system"""

# set the hyperparameters
T_length = 200
X0 = np.array((0.0, 0.0, 0.0))
#pdb.set_trace()
# T = np.array((0.0, 1))
# define the batch system
def state_update(t, x, u, params):
    # get the parameter from the params
    batch_num = params.get('batch_num', 0)
    # Parameter setup
    # pdb.set_trace()
    sigma1 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
    sigma2 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
    sigma3 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
    sigma4 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
    sigma1+=0.5*np.sin(0.1*t)
    sigma2 +=0.5*np.sin(0.1*t)
    sigma3 += 0.5*np.sin(0.1*t)
    sigma4 += 0.5*np.sin(0.1*t)
    # pdb.set_trace()
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3 = np.array([x[2]])
    # Compute the discrete updates
    dz1 = (1.607 + 0.0804 * sigma1) * z1 - (0.6086 + 0.0304 * sigma2) * z2 - (0.9282 + 0.0464 * sigma3) * z3 + (
            1.239 + 0.062 * sigma4) * u
    dz2 = z1
    dz3 = u
    # pdb.set_trace()
    return [dz1, dz2, dz3]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y = x[0]

    return [y]


batch_system = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2', 'dz3'), dt=1, name='Linear_injection_modeling')
#controlled_system = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0)


args.env = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0,action_co=10)
args.env_eval = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0,action_co=10)

# Hyperparameters
args.agent.cri_target = True
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
#args.reward_scale = 2 ** -10000  # RewardRange: -1800 < -200 < -50 < 0
#args.gamma = 0.99
args.gamma = 0.99
#args.net_dim = 2 ** 6
#args.net_dim = 2 ** 5000
args.net_dim = 2 ** 8
#args.net_dim = 2 ** 9
args.batch_size = args.net_dim * 2
#args.batch_size = args.net_dim
#args.batch_size =  N
args.target_step = 3*T_length
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
#args.learning_rate=9e-50000
args.eval_gap=2 ** 6
args.eval_times1 = 20
args.eval_times2 = 60
#args.eval_gap=2 ** 2
args.max_memo=200*2000
#args.learning_rate=0.001
##############################################
'cwd'
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
args.cwd = os.path.join(current_dir, 'runs_AgentModSAC_Linear_injection_modeling/1')


#pprint.pprint(args.__dict__)
#pdb.set_trace()
train_and_evaluate(args)