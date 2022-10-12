from net import ActorSAC
import pdb
import os
import control
from ilcenv import BatchSysEnv
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
import csv
"""create batch system"""

# set the hyperparameters
# define the dimensions of the state space
m=3
n=1
r=1# No useful
l=1
batch=10
T_length = 200
X0 = np.array((0.0, 0.0, 0.0))
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
    states=('dz1', 'dz2', 'dz3'), dt=1, name='ILCsystem')
sys = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0)

"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
#pdb.set_trace()
data_dir=os.path.join(current_dir, "runs2/10")
#data_dir=os.path.join(current_dir, "runs2/1")
model_dir=os.path.join(data_dir, "actor.pth")
"""create the actor and load the trained model"""
mid_dim=2 ** 8
test_model=ActorSAC(mid_dim, sys.state_dim, sys.action_dim)
#pdb.set_trace()
model_dict=test_model.load_state_dict(torch.load(model_dir))
test_model.eval()
#pdb.set_trace()
y_data_list=[]
u_ilc_data_list=[]
u_rl_data_list=[]
"""numpy"""
y_data=np.zeros((T_length,l))
u_ilc_data=np.zeros((T_length,n))
u_rl_data=np.zeros((T_length,n))
state = sys.reset()
#pdb.set_trace()
for batch_num in range(batch):
    for episode_step in range(T_length):
        # pdb.set_trace()
        s_tensor = torch.as_tensor((state,), dtype=torch.float32)
        # pdb.set_trace()
        a_tensor = test_model(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        #pdb.set_trace()
        u_ilc_data[episode_step][0]=(state[13])
        u_rl_data[episode_step][0] = (state[0])
        state, reward, done, _ = sys.step(action)
        #u_rl_data[episode_step][0]=np.float64(action)
        y_data[episode_step]=state[6]
        #pdb.set_trace()
        """
        if episode_step==(T_length-1):
            pdb.set_trace()
        """
        if done:
            break
    #pdb.set_trace()
    state = sys.reset()
    """put one batch information to the list"""
    y_data_list.append(copy.deepcopy(y_data))
    u_ilc_data_list.append(copy.deepcopy(u_ilc_data))
    u_rl_data_list.append(copy.deepcopy(u_rl_data))


#pdb.set_trace()
"""Plot the 3d visibale figure"""
"""define the reference trajectory"""
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
#y_ref=200*np.ones((1,T_length))
y_ref=200*np.ones((T_length,1))
#pdb.set_trace()
y_ref[100:]=1.5*y_ref[100:]
t=range(T_length)
legend_list=[]
#1.response
fig=plt.figure()
for item in range(batch):
    plt.plot(t, y_data_list[item], linewidth=1)
    legend_list.append(item)
xlable = 'Time:t'
ylable = 'Output:$y_{k}(t)$'

plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(legend_list)
plt.show()


#pdb.set_trace()
#2. RL control signal
fig_control=plt.figure()
for item in range(batch):
    plt.plot(t, u_rl_data_list[item], linewidth=1)
    #legend_list.append(item)
xlable = 'Time:t'
ylable = 'RL control:$u_{k}^{rl}(t)$'

plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(legend_list)
plt.show()

#pdb.set_trace()

#3. ILC control signal
fig_control=plt.figure()
for item in range(batch):
    plt.plot(t, u_ilc_data_list[item], linewidth=1)
    #legend_list.append(item)
xlable = 'Time:t'
ylable = 'ILC control:$u_{k}^{ILC}(t)$'

plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(legend_list)
plt.show()

pdb.set_trace()

"""
#4.SAE

#pdb.set_trace()
SAE=np.zeros(batch)
for batch_index_2d in range(batch):
    y_out_time = y_data_list[batch_index_2d]
    #pdb.set_trace()
    for time in range(T_length):
        SAE[batch_index_2d] += (abs(y_out_time[time] - y_ref[time])) ** 2
    SAE[batch_index_2d] = math.sqrt(SAE[batch_index_2d] / T_length)
plt.figure()
batch_time=range(batch)
x_major_locator=MultipleLocator(1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE,linewidth=1.5,color='red',linestyle='--')
#plt.plot(batch_time,SAE,linewidth=2)
#pdb.set_trace()
#plt.plot(batch_time,y_ILC_show,linewidth=1,color='red')
#plt.plot(batch_time,y_ILC_show,linewidth=1.5,color='black',linestyle=':')

plt.grid()
xlable = 'Batch:k '
ylable = 'Root Mean Squared Error (RMSE)'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2DILC_RL','2DILC'])
#plt.savefig('batch_RMES_action10.png',dpi=700)
plt.show()
"""


pdb.set_trace()
a=2

