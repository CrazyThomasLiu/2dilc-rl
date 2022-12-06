from run import *
from agent import AgentModSAC,AgentSAC
import pdb
import control
#from cstr_MIMO.env_MIMO_cstr_40_time_batch_constrain150 import BatchSysEnv
from cstr_MIMO.env_MIMO_cstr_30_time_batch_constrain1000_n1000 import BatchSysEnv
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
args.break_step = 200*32000
#args.agent.lambda_entropy = 0.04
#args.agent.lambda_entropy = 0.08
#print(torch.get_default_dtype())
#pdb.set_trace()

"""create batch system"""

# set the hyperparameters
# define the batch system
def state_update(t, x, u, params):
    batch_num = params.get('batch_num', 0)
    # Map the states into local variable names
    batch_num = params.get('batch_num', 0)
    # print(batch_num)
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1 = np.array([u[0]])
    n2 = np.array([u[1]])
    # Compute the discrete updates
    a = 1 + 0.1 * np.sin(2.5 * t * np.pi) + 0.1 * np.sin(batch_num * np.pi / 10)
    # a = 1+ 0.1 * np.sin(batch_num * np.pi / 10)
    dz1 = -(a + 7.2 * np.power(10., 10) * np.exp(-np.power(10., 4) / z2)) * z1 + n1
    # dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - a * z2 + 0.041841 * n2 + 310 * a
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y1 = x[0]
    y2 = x[1]

    return [y1, y2]


Nonlinear_CSTR = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u1', 'u2'), outputs=('y1', 'y2'),
    states=('dz1', 'dz2'), dt=0, name='Nonlinear_CSTR')

print("Continuous system:", control.isctime(Nonlinear_CSTR))

"define the initial state "
X0 = np.array((0.47, 396.9))  # Initial x1, x2
T = np.array((0.0, 0.01))
sample_time = 0.01
T_length = 200
x_k = copy.deepcopy(np.expand_dims(X0, axis=0))
args.env = BatchSysEnv(T_length=T_length, sys=Nonlinear_CSTR, X0=X0)
args.env_eval = BatchSysEnv(T_length=T_length, sys=Nonlinear_CSTR, X0=X0)
#pdb.set_trace()

args.env.target_return = 200000
args.env_eval.target_return = 200000
# Hyperparameters
args.agent.cri_target = True
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
#args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
#args.gamma = 0.99
args.gamma = 0.99
#args.net_dim = 2 ** 6
#args.net_dim = 2 ** 7
args.net_dim = 2 ** 8
#args.net_dim = 2 ** 9
args.batch_size = args.net_dim * 2
#args.batch_size = args.net_dim
#args.batch_size =  N
args.target_step = 20*T_length
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
#args.learning_rate=9e-4
args.eval_gap=2 ** 6
args.eval_times1 = 20
args.eval_times2 = 60
#args.eval_gap=2 ** 2
args.max_memo=200*2000
#args.learning_rate=0.001
##############################################
#pprint.pprint(args.__dict__)
#pdb.set_trace()
train_and_evaluate(args)