import sys
import os
import pdb
from net import ActorSAC
import os
import control
from env_nonlinear_batch_reactor import BatchSysEnv
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
save_csv=True
X0 = np.array((0.5, 310.))
# define the batch system
def state_update(t, x, u, params):
    batch_num = params.get('batch_num', 0)
    # Compute the discrete updates
    a=1+1*np.sin(2.5*t* np.pi)+1*np.sin(batch_num * np.pi / 10)
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1=np.array([u[0]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+0.5
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2+0.041841*n1 +310*a
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
model_dir=os.path.join(current_dir, "best_actor.pth")
"""create the actor and load the trained model"""
mid_dim=2 ** 8
test_model=ActorSAC(mid_dim, sys.state_dim, sys.action_dim)
model_dict=test_model.load_state_dict(torch.load(model_dir))
sys.save_or_load_history_implementation(current_dir, if_save=False)
y_data_list=[]
u_ilc_data_list=[]
u_rl_data_list=[]
"""numpy"""
y_data=np.zeros((T_length,l))
u_ilc_data=np.zeros((T_length,n))
u_rl_data=np.zeros((T_length,n))
state = sys.reset()
for batch_num in range(batch):
    for episode_step in range(T_length):
        s_tensor = torch.as_tensor((state,), dtype=torch.float32)
        a_tensor = test_model(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        u_ilc_data[episode_step][0]=(100*state[13])
        state, reward, done, _ = sys.step(action)
        u_rl_data[episode_step][0] = (100*state[0])
        y_data[episode_step]=state[6]
        if done:
            break
    state = sys.reset()
    """put one batch information to the list"""
    y_data_list.append(copy.deepcopy(y_data))
    u_ilc_data_list.append(copy.deepcopy(u_ilc_data))
    u_rl_data_list.append(copy.deepcopy(u_rl_data))

'Save the all data in csv'
if save_csv==True:
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
t=0.01*np.arange(T_length)
batch_before=np.ones(T_length,dtype=int)
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
ax.view_init(31, -42)
if save_figure==True:
    plt.savefig('Nonlinear_batch_reactor_output.pdf')
    plt.savefig('Nonlinear_batch_reactor_output.jpg')


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
    plt.savefig('Nonlinear_batch_reactor_rl_input.pdf')

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
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_ilc_data_list[item2].squeeze(),linewidth=1,color='black')

ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('Nonlinear_batch_reactor_ilc_input.pdf')
plt.show()


#4.SAE

SAE=np.zeros(batch)
for batch_index_2d in range(batch):
    y_out_time = y_data_list[batch_index_2d]
    for time in range(T_length):
        SAE[batch_index_2d] += (abs(y_out_time[time] - y_ref[time])) ** 2
    SAE[batch_index_2d] = math.sqrt(SAE[batch_index_2d] / T_length)

"""save the sqrt to the csv"""
if save_csv== True:
    with open('2DILC-RL_RMSE.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Value'])
        writer.writerows(map(lambda x: [x], SAE))
# Draw the control performance of the pure 2D ILC controller and 2D ILC-RL control scheme
batch_rmse=50
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the 2dilc rmse"""
ILC_dir=os.path.join(current_dir, "2DILC_RMSE_nonlinear_batch_reactor.csv")
f_ILC=open(ILC_dir,'r')
num=0
y_2dilc=np.zeros(batch_rmse)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        y_2dilc[num]=row['Value']
        num+=1

"""2. Load the 2dilc_rl rmse"""
rl_dir=os.path.join(current_dir, "2DILC-RL_RMSE.csv")
length=batch_rmse+1
f_rl=open(rl_dir,'r')
t=np.zeros(length)
num=0
y_rl=np.zeros(length)
with f_rl:
    reader=csv.DictReader(f_rl)
    for row in reader:
        y_rl[num]=row['Value']
        num+=1


# reduce the length of the rmse
y_2dilc_show=y_2dilc[0:batch_rmse]
y_2dilc_rl_show=y_rl[0:batch_rmse]
"2. Plot of the sum RMSE"
batch_time=range(1,batch_rmse+1)
fig=plt.figure(figsize=(7,5.5))
x_major_locator=MultipleLocator(int(batch_rmse/10))
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_2dilc_show,linewidth=1.5,color='tab:blue',linestyle = 'dashdot')
plt.plot(batch_time,y_2dilc_rl_show,linewidth=1.5,color='tab:orange',linestyle='solid')
plt.grid()

xlable = 'Batch:$\mathit{k} $'
ylable = 'RMSE:$\mathit{I_{k}}$'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D Iterative Learning Control Scheme','2D ILC-RL Control Scheme'])
if save_figure==True:
    plt.savefig('Nonlinear_batch_reactor_compare_rmse.pdf')

plt.show()

# Draw the 2D DRL compensation signal at t=150 for all batches.
batch=50
time_length=200
length=15000
time_section=149
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the rl input"""
rl_dir=os.path.join(current_dir, "Nonlinear_batch_reactor_rl_input.csv")
f = open(rl_dir, 'r')
t=np.zeros(length)
num=0
rl_input=np.zeros(length)
with f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        for item in row:
            rl_input[num]=float(item)
            num+=1

"""2. set the time transaction"""
rl_input_show=np.zeros(batch)
for item in range(batch):
    rl_input_show[item]=rl_input[300*item+time_section]
if save_csv== True:
    with open('compensation_input_time150.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Compensation'])
        writer.writerows(map(lambda x: [x],  rl_input_show))
"3. Plot of time transaction"
batch_time=range(1,batch+1)
fig=plt.figure(figsize=(9.0,5.5))
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
x_major_locator=MultipleLocator(int(batch/10))

plt.plot(batch_time,rl_input_show,linewidth=1.5,color='black',linestyle='solid')
plt.grid()
# plot the sin function
xlable = 'Batch:$\mathit{k} $'
ylable = 'DRL Compensation Signal:$\mathit{u_{k,150}}$'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
labels=ax.get_xticklabels()+ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
plt.tick_params(axis='both',width=1.5,length=5)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

if save_figure==True:
    plt.savefig('Nonlinear_batch_reactor_rl_input_time150.pdf')

plt.show()

