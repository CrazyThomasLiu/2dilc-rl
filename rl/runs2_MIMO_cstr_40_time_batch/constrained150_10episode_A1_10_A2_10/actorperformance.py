import sys
import os
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',2)[0]
sys.path.append(config_path)
import pdb
from net import ActorSAC
import pdb
import os
import control
#from ilcenv_ilcinrl_time_batch_sinus import BatchSysEnv
from cstr_MIMO.env_MIMO_cstr_40_time_batch_constrain150 import BatchSysEnv
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
"1. define the CSTR nonlinear model"
# define the dimensions of the state space
m = 2  # dimension of the state
n = 2  # input
# r=2# No useful
l = 2  # output
T_length = 200
# T_length=300
batch=100
batch_SAE=100
save_figure = True


def state_update(t, x, u, params):
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

"2. define the initial state "
X0 = np.array((0.47, 396.9))  # Initial x1, x2
T = np.array((0.0, 0.01))
sample_time = 0.01
x_k = copy.deepcopy(np.expand_dims(X0, axis=0))
controlled_system = BatchSysEnv(T_length=T_length, sys=Nonlinear_CSTR, X0=X0)
"3. Load the weight of the actor"
"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
#pdb.set_trace()
#data_dir=os.path.join(current_dir, "runs2/10")
model_dir=os.path.join(current_dir, "actor.pth")



"""create the actor and load the trained model"""
mid_dim=2 ** 8
test_model=ActorSAC(mid_dim, controlled_system.state_dim, controlled_system.action_dim)
#pdb.set_trace()
model_dict=test_model.load_state_dict(torch.load(model_dir))
#pdb.set_trace()





"4. simulation"
y_batch = np.zeros((T_length,n))
y_data = []
u_ilc_batch = np.zeros((T_length,n))
u_ilc_data = []

u_rl_batch = np.zeros((T_length,n))
u_rl_data = []
state = controlled_system.reset()

#pdb.set_trace()
for batch_index in range(batch):
    for item in range(T_length):
        s_tensor = torch.as_tensor((state,), dtype=torch.float32)
        # pdb.set_trace()
        a_tensor = test_model(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        #pdb.set_trace()
        state, out_tem, done, invalid = controlled_system.step(action)
        # pdb.set_trace()
        y_batch[item, 0] = state[6]
        y_batch[item, 1] = state[20]
        u_ilc_batch[item, 0] = state[3]
        u_ilc_batch[item, 1] = state[17]
        u_rl_batch[item, 0] = state[0]
        u_rl_batch[item, 1] = state[14]
    #pdb.set_trace()
    y_data.append(y_batch)
    u_ilc_data.append(u_ilc_batch)
    u_rl_data.append(u_rl_batch)
    # pdb.set_trace()
    state=controlled_system.reset()
    y_batch = np.zeros((T_length,n))
    u_ilc_batch = np.zeros((T_length, n))
    u_rl_batch = np.zeros((T_length, n))
#pdb.set_trace()


"""Plot the 3d visibale figure """

#define the reference trajectory
y_ref=np.ones((T_length,2))
y_ref[:,0]=0.57*y_ref[:,0]
y_ref[:,1]=395*y_ref[:,1]
#pdb.set_trace()
"1.Output Response"
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
"1.1 y1 production concentration"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
#pdb.set_trace()
#ax.plot3D(batch_before,t, y_data[0].squeeze(),linewidth=1,color='black')
ax.plot3D(Batch_length*(batch-1),t, y_ref[:,0].squeeze(),linestyle='dashed',linewidth=2,color='royalblue')
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, y_data[item2][:,0].squeeze(),linewidth=1,color='black')

#pdb.set_trace()
xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Output:Production Concentration $kmol/m^{3}$'
zlable = '$y_{1}$:Production Concentration $kmol/m^{3}$'
plt.title('Output Response: Production Concentration')
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
ax.legend(['y_Ref','y_out'])
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('Output Response_y1.png',dpi=700)
#plt.show()
#pdb.set_trace()


"1.2 y2  Temperature"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
#pdb.set_trace()
#ax.plot3D(batch_before,t, y_data[0].squeeze(),linewidth=1,color='black')
ax.plot3D(Batch_length*(batch-1),t, y_ref[:,1].squeeze(),linestyle='dashed',linewidth=2,color='royalblue')
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, y_data[item2][:,1].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Output:Temperature $K$'
zlable = '$y_{2}$:Temperature $K$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
plt.title('Output Response: Temperature')
ax.legend(['y_Ref','y_out'])
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('Output Response_y2.png',dpi=700)
plt.show()
#pdb.set_trace()

"2.Input Signal"
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }

"2.1 ILC Control Signal"
"2.1.1 u1 Feed Concentration"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_ilc_data[item2][:,0].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Feed Concentration $kmol/m^{3}$'
zlable = '$u{1}$:Feed Concentration $kmol/m^{3}$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
plt.title('Input Signal: ILC-Feed Concentration')
#ax.legend(['y_Ref','y_out'])
ax.view_init(52, -16)
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('ILC Control Signal_u1.png',dpi=700)
#plt.show()
#pdb.set_trace()

"2.1.2 u2 Heat"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_ilc_data[item2][:,1].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Heat $kJ/min$'
zlable = '$u_{2}$:Heat $kJ/min$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
plt.title('Input Signal: ILC-Cooler')
#ax.legend(['y_Ref','y_out'])
ax.view_init(52, -16)
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('ILC Control Signal_u2.png',dpi=700)
plt.show()
#pdb.set_trace()
"2.2 RL Control Signal"
"2.2.1 u1 Feed Concentration"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_rl_data[item2][:,0].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Feed Concentration $kmol/m^{3}$'
zlable = '$u{1}$:Feed Concentration $kmol/m^{3}$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
plt.title('Input Signal: RL-Feed Concentration')
#ax.legend(['y_Ref','y_out'])
ax.view_init(52, -16)
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('RL Control Signal_u1.png',dpi=700)
#plt.show()
#pdb.set_trace()

"2.2.2 u2 Heat"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_rl_data[item2][:,1].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Heat $kJ/min$'
zlable = '$u_{2}$:Heat $kJ/min$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
plt.title('Input Signal: RL-Cooler')
#ax.legend(['y_Ref','y_out'])
ax.view_init(52, -16)
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('RL Control Signal_u2.png',dpi=700)
plt.show()




pdb.set_trace()
"3.SAE"
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
"3.1 Calculation of the SAE"
SAE_y1=np.zeros(batch_SAE)
SAE_y2=np.zeros(batch_SAE)
SAE_sum=np.zeros(batch_SAE)
for batch_index in range(batch_SAE):
    y_list_y1 = y_data[batch_index][:,0]
    y_list_y2 = y_data[batch_index][:, 1]
    #pdb.set_trace()
    for time in range(T_length):
        SAE_y1[batch_index] += (abs(y_list_y1[time] - y_ref[time,0])) ** 2
        SAE_y2[batch_index] += (abs(y_list_y2[time] - y_ref[time, 1])) ** 2
    SAE_y1[batch_index] = math.sqrt(SAE_y1[batch_index] / T_length)
    SAE_y2[batch_index] = math.sqrt(SAE_y2[batch_index] / T_length)
    SAE_sum[batch_index]=SAE_y1[batch_index]+SAE_y2[batch_index]
#plt.figure()
#pdb.set_trace()
"""3.2 Load the sqrt form the only ILC"""
ILCsac_dir=os.path.join(current_dir, "constrain_0_n150_y1.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_y1=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_y1[num]=row['Value']
        num+=1
ILCsac_dir=os.path.join(current_dir, "constrain_0_n150_y2.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_y2=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_y2[num]=row['Value']
        num+=1

ILCsac_dir=os.path.join(current_dir, "constrain_0_n150_sum.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_sum=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_sum[num]=row['Value']
        num+=1

#pdb.set_trace()
"3.3 Plot of the y1"
plt.subplot(2,1,1)
batch_time=range(1,batch_SAE+1)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_y1,linewidth=2,color='blue',linestyle=':')
plt.plot(batch_time,y_ILC_y1,linewidth=3,color='red',linestyle=':')
#plt.plot(batch_time,SAE_tem,linewidth=3,color='red',linestyle=':')
plt.grid()
xlable = 'Batch:k'
#ylable = 'Root Mean Squared Error (RMSE)'
ylable = 'RMSE'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D-ILC-RL','2D-ILC'])
plt.title('Production Concentration')
"3.4 Plot of the y2"

plt.subplot(2,1,2)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_y2,linewidth=2,color='blue',linestyle=':')
plt.plot(batch_time,y_ILC_y2,linewidth=3,color='red',linestyle=':')
plt.grid()
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D-ILC-RL','2D-ILC'])
plt.title('Temperature')
if save_figure==True:
    plt.savefig('SAE.png',dpi=700)

"3.5 Plot of the sum SAE"

plt.figure()
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_sum,linewidth=2,color='blue',linestyle=':')
plt.plot(batch_time,y_ILC_sum,linewidth=3,color='red',linestyle=':')
plt.grid()
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D_ILC_RL','ILC'])
if save_figure==True:
    plt.savefig('constrain_0_n150_total_SAE.png',dpi=700)



plt.show()
pdb.set_trace()
a=2

