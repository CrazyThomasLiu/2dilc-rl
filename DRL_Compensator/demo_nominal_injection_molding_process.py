from run_for_implementation import *
from agent import AgentModSAC
import pdb
import control
import sys
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',1)[0]
sys.path.append(config_path)
from env_sys.env_linear_injection_modeling import BatchSysEnv
import numpy as np
import os
# create Agent
args = Arguments(if_on_policy=False)
args.agent =AgentModSAC()
args.break_step = 200*4000
"""create batch system"""

# set the hyperparameters
T_length = 200
X0 = np.array((0.0, 0.0, 0.0))
#pdb.set_trace()
# T = np.array((0.0, 1))
# define the batch system
def state_update(t, x, u, params):
    # get the parameter from the params
    # pdb.set_trace()
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3 = np.array([x[2]])
    # Compute the discrete updates
    dz1 = 1.607 * z1 - 0.6086* z2 - 0.9282* z3 + 1.239 * u
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

train_and_evaluate(args)