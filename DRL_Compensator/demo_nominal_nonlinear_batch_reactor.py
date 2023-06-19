from run_for_nominal_nonlinear_batch_reactor import *
from agent import AgentModSAC
import pdb
import control
import os
import sys
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',1)[0]
sys.path.append(config_path)
from env_sys.env_nonlinear_batch_reactor import BatchSysEnv
import numpy as np
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
import copy
# create Agent
args = Arguments(if_on_policy=False)
args.agent =AgentModSAC()
#args.agent =AgentSAC()
#pdb.set_trace()
#args.agent.if_use_gae = True
#args.break_step = 300*10000
args.break_step = 300*10000
#args.agent.lambda_entropy = 0.04
#args.agent.lambda_entropy = 0.08
#print(torch.get_default_dtype())
#pdb.set_trace()

"""create batch system"""

# set the hyperparameters
T_length = 300
X0 = np.array((0.5, 310.))
#pdb.set_trace()
# T = np.array((0.0, 1))
# define the batch system
def state_update(t, x, u, params):
    batch_num = params.get('batch_num', 0)
    # Compute the discrete updates
    #a=1+1*np.sin(2.5*t* np.pi)+1*np.sin(batch_num * np.pi / 10)
    a=1.
    #a = 1 + 0.5 * np.sin(2.5 * t * np.pi) + 0.5 * np.sin(batch_num * np.pi / 10)
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1=np.array([u[0]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+0.5
    #dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 50000) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2+0.041841*n1 +310*a
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y1 = x[1]

    return [y1]


batch_system = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u1'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=0, name='SISO_CSTR')
#controlled_system = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0)


args.env = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0)
args.env_eval = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0)

# Hyperparameters
args.agent.cri_target = True
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
args.gamma = 0.99
args.net_dim = 2 ** 8
args.batch_size = args.net_dim * 2
args.target_step = 10*T_length
"""
author:jianan liu
"""
args.repeat_times=1
args.soft_update_tau = 2 ** -8
args.learning_rate=3e-4
args.eval_gap=2 ** 6
args.eval_times1 = 20
args.eval_times2 = 60
args.max_memo=300*2000
'cwd'
train_and_evaluate(args)