import sys
import os
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',2)[0]
sys.path.append(config_path)
config_path_rl='/home/liu/deeplearn/2dilc_rl/implementation/rl'
sys.path.append(config_path_rl)
import pdb
from net import ActorSAC
import pdb
import os
import control
from env_SISO_CSTR import BatchSysEnv
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
m=2
n=1
r=1# No useful
l=1
batch=50
T_length = 300
save_figure=True
save_csv=False
X0 = np.array((0.5, 310.))
#pdb.set_trace()
# T = np.array((0.0, 1))
# define the batch system
def state_update(t, x, u, params):
    batch_num = params.get('batch_num', 0)
    # Compute the discrete updates
    a=1+1*np.sin(2.5*t* np.pi)+1*np.sin(batch_num * np.pi / 10)
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
sys = BatchSysEnv(T_length=T_length, sys=batch_system, X0=X0,action_co=10000.)

"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
#pdb.set_trace()
#data_dir=os.path.join(current_dir, "runs2/10")
model_dir=os.path.join(current_dir, "best_actor.pth")

"""create the actor and load the trained model"""
mid_dim=2 ** 8
test_model=ActorSAC(mid_dim, sys.state_dim, sys.action_dim)
#pdb.set_trace()
model_dict=test_model.load_state_dict(torch.load(model_dir))
#pdb.set_trace()
sys.save_or_load_history_implementation(current_dir, if_save=False)
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
        u_ilc_data[episode_step][0]=(100*state[13])
        #u_rl_data[episode_step][0] = (state[0])
        state, reward, done, _ = sys.step(action)
        u_rl_data[episode_step][0] = (100*state[0])
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
pdb.set_trace()
#np.savetxt('myfile.csv', u_ilc_data, delimiter=',')
'Save the all data in csv'
#1 ILC input
f1 = open('Nonlinear_batch_reactor_ilc_input.csv', 'w')
with f1:
    writer = csv.writer(f1)
    for number in u_ilc_data_list:
        for row in number:
            writer.writerow(row)
#2 RL input
f2 = open('Nonlinear_batch_reactor_rl_input.csv', 'w')
with f2:
    writer = csv.writer(f2)
    for number in u_rl_data_list:
        for row in number:
            writer.writerow(row)
#3 Output Response
f3 = open('Nonlinear_batch_reactor_output.csv', 'w')
with f3:
    writer = csv.writer(f3)
    for number in y_data_list:
        for row in number:
            writer.writerow(row)
#pdb.set_trace()
#pdb.set_trace()
"""Plot the 3d visibale figure"""
"""define the reference trajectory"""
#y_ref=200*np.ones((1,T_length))
y_ref = np.ones((T_length, 1))
y_ref[0:100, 0] = 370 * y_ref[0:100, 0]
y_ref[100:T_length, 0] = 350 * y_ref[100:T_length, 0]


font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 14
         }

#1.response
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#pdb.set_trace()
t=0.01*np.arange(T_length)
batch_before=np.ones(T_length,dtype=int)
#pdb.set_trace()
#ax.plot3D(batch_before,t, y_data[0].squeeze(),linewidth=1,color='black')
ax.plot3D(batch_before*(batch-1),t, y_ref.squeeze(),linestyle='dashed',linewidth=2,color='tab:green')
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, y_data_list[item2].squeeze(),linewidth=1,color='black')


xlable = 'Batch:$k$'
ylable = 'Time:min'
zlable = 'Output Response'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
ax.legend(['$y_{k,t}^{r}$','$y_{k,t}$'])
#ax.view_init(52, -16)
#plt.savefig('3DOut.png',dpi=700)
ax.view_init(31, -42)
if save_figure==True:
    #plt.savefig('Nonlinear_batch_reactor_output.png',dpi=900)
    plt.savefig('Nonlinear_batch_reactor_output.pdf')
#plt.show()
#pdb.set_trace()



#2. RL control signal
fig_control=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_rl_data_list[item2].squeeze(),linewidth=1,color='black')

xlable = 'Batch:$k$'
ylable = 'Time:min'
zlable = 'DRL control signal'
#change the font3

font3 = {'family': 'Arial',
         'weight': 'bold',
         'size': 10
         }
ax.set_xlabel(xlable,font3)
ax.set_ylabel(ylable,font3)
ax.set_zlabel(zlable,font3)
ax.view_init(40, -19)
if save_figure==True:
    #plt.savefig('Nonlinear_batch_reactor_rl_input.png',dpi=700)
    plt.savefig('Nonlinear_batch_reactor_rl_input.pdf')
#plt.show()
#pdb.set_trace()

#3. ILC control signal
fig_control=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
xlable = 'Batch:$k$'
ylable = 'Time:min'
zlable = 'ILC control signal'
ax.set_xlabel(xlable,font3)
ax.set_ylabel(ylable,font3)
ax.set_zlabel(zlable,font3)
#ax.legend(['$y_{k,t}^{r}$','$y_{k,t}$'])
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_ilc_data_list[item2].squeeze(),linewidth=1,color='black')

ax.view_init(40, -19)
if save_figure==True:
    #plt.savefig('Nonlinear_batch_reactor_ilc_input.png',dpi=700)
    plt.savefig('Nonlinear_batch_reactor_ilc_input.pdf')
plt.show()
pdb.set_trace()



#4.SAE

#pdb.set_trace()
SAE=np.zeros(batch)
#pdb.set_trace()
for batch_index_2d in range(batch):
    y_out_time = y_data_list[batch_index_2d]
    #if batch_index_2d==49:
        #print(y_out_time)
        #pdb.set_trace()
    for time in range(T_length):
        SAE[batch_index_2d] += (abs(y_out_time[time] - y_ref[time])) ** 2
    SAE[batch_index_2d] = math.sqrt(SAE[batch_index_2d] / T_length)
    #SAE[batch_index_2d] = math.log10(SAE[batch_index_2d])

"""save the sqrt to the csv"""
if save_csv== True:
    with open('2DILC-RL_training_finished.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(map(lambda x: [x], SAE))
#pdb.set_trace()
plt.figure()
batch_time=range(batch)
x_major_locator=MultipleLocator(2)
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
plt.ylim((0, 35))
#ylable = '$log_{10}RMSE$'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2DILC_RL','2DILC'])
#plt.savefig('batch_RMES_action10_log10.png',dpi=700)
if save_figure==True:
    #plt.savefig('batch_RMES.png',dpi=700)
    plt.savefig('batch_RMES.pdf')
plt.show()



pdb.set_trace()
a=2

