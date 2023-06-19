from run_for_implementation_practical import *
from agent_practical import AgentModSAC
import pdb
import control
import sys
import os
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',1)[0]
sys.path.append(config_path)
from env_sys.env_linear_injection_modeling import BatchSysEnv
import numpy as np
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
import copy
import os
# create Agent
args = Arguments(if_on_policy=False)
args.agent =AgentModSAC()
args.break_step = 200*2000

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

args.env = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0,action_co=10)
args.env_eval = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0,action_co=10)

# Hyperparameters
args.agent.cri_target = True
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
args.gamma = 0.99
args.net_dim = 2 ** 8
args.batch_size = args.net_dim * 2
args.target_step = 3*T_length
"""
author:jianan liu
"""
args.repeat_times=1
args.soft_update_tau = 2 ** -8
args.learning_rate=3e-4
args.eval_gap=2 ** 6
args.eval_times1 = 20
args.eval_times2 = 60
args.max_memo=200*2000
'cwd'
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""the path where the pretrained data with the virtual environment is loaded"""
args.cwd = os.path.join(current_dir, 'runs_AgentModSAC_Linear_injection_modeling_practical/1')
train_and_evaluate(args)