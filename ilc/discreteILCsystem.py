import control
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import copy
import math
#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
import csv
# define the dimensions of the state space
m=3
n=1
r=1# No useful
l=1
T_length=200
#T_length=20
batch=20
def state_update(t, x, u, params):
    # Parameter setup

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3 = np.array([x[2]])
    # Compute the discrete updates
    dz1=1.607*z1-0.6086*z2-0.9282*z3+1.239*u
    dz2=z1
    dz3 = u
    #pdb.set_trace()
    return [dz1, dz2,dz3]
def ouput_update(t, x, u, params):
    # Parameter setup


    # Compute the discrete updates
    y=x[0]

    return [y]
io_nonlinearsystem = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2', 'dz3'),dt=1,name='ILCsystem')
#pdb.set_trace()
#X0 = [0.0, 0.0,0.0]
X0 = np.array((0.0,0.0,0.0))
T = np.array((0.0,1))
#T=np.linspace(0, 3.5,36)
input=np.array((1.,1.))
#pdb.set_trace()

#define the 2D systems
#y_ref=200*np.ones((1,T_length))
y_ref=200*np.ones((T_length,1))
#pdb.set_trace()
y_ref[100:]=1.5*y_ref[100:]
x_k=np.zeros((1,m))
x_k_last=np.zeros((200+1,m))
x_k_current=np.zeros((200+1,m))
y_k_last=np.zeros((200,l))
sigma_k=np.zeros((1,m))
#sigma_k=x_k[0]-x_k_last[0]
e_k=np.zeros((1,l))
#merge the sigma_k and e_k
x_2d=np.zeros((1,l+m))
K=np.array([[-1.4083788,0.57543156,0.87756631,0.71898388]])
#K=np.array([[-1.4201,0.58403,0.89073,0.70219]])
r_k=np.zeros((1,n))
r_k=np.zeros((1,n))
u_k=np.zeros((1,n))
u_k_last=np.zeros((T_length,n))
#pdb.set_trace()
#define the output data
y_data=[]
u_data=[]


# Simulate the system
#t, y,x = control.input_output_response(io_nonlinearsystem, T, input, X0,return_x=True)
#t, y = control.input_output_response(io_nonlinearsystem, T, 1, X0)


for batch_index in range(batch):
    x_k=np.zeros((1,m))
    X0 = np.array((0.0, 0.0, 0.0))
    #pdb.set_trace()
    for item in range(T_length):
        #if item ==0:
        #pdb.set_trace()
        tem_x=x_k[0]-x_k_last[item]  # 上一个批次应该是0
        tem_y=y_ref[item]-y_k_last[item]
        x_2d=np.block([[tem_x,tem_y]])
        #pdb.set_trace()
        r_k[0][0]=K@x_2d.T
        u_k[0][0]=u_k_last[item][0]+r_k[0][0]
        input[0]=u_k[0][0]
        input[1] = u_k[0][0]
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(io_nonlinearsystem, T, input, X0=X0, return_x=True)
        #pdb.set_trace()
        # change the initial state
        X0[0] = x_step[0][1]
        X0[1] = x_step[1][1]
        X0[2] = x_step[2][1]
        # save the data into the memory
        u_k_last[item][0]=u_k[0][0]
        y_k_last[item]=y_step[1]
        #pdb.set_trace()
        for item1 in range(m):
            x_k_current[(item+1)][item1]=x_step[item1][1]
            x_k[0][item1]=x_step[item1][1]   #change the current information
        #x_k_last[item]=x_step[1]
        #print(y_step[1])
        #pdb.set_trace()
    x_k_last=copy.deepcopy(x_k_current)
    #pdb.set_trace()
    #deep copy
    y_data.append(copy.deepcopy(y_k_last))
    u_data.append(copy.deepcopy(u_k_last))
#pdb.set_trace()
# Plot the 3d visibale figure
#1.response
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
batch_before=np.ones(T_length,dtype=int)
#pdb.set_trace()
#ax.plot3D(batch_before,t, y_data[0].squeeze(),linewidth=1,color='black')
ax.plot3D(batch_before*(19),t, y_ref.squeeze(),linestyle='dashed',linewidth=2,color='royalblue')
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, y_data[item2].squeeze(),linewidth=1,color='black')


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'Batch:k'
ylable = 'Time:t'
zlable = 'Output:$y_{k}(t)$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
#ax.set_zlabel(zlable,font2)
ax.legend(['y_Ref','y_out'])
#ax.view_init(52, -16)
#plt.savefig('3DOut.png',dpi=700)

plt.show()
#2. control signal
fig_control=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for item2 in range(batch):
    batch_plot=batch_before*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_data[item2].squeeze(),linewidth=1,color='black')

ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
plt.show()
#3.SAE
SAE=np.zeros(batch)
for batch_index_2d in range(batch):
    y_out_time = y_data[batch_index_2d]
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
plt.plot(batch_time,SAE,linewidth=3,color='black',linestyle=':')

plt.grid()
xlable = 'Batch:k '
ylable = 'Root Mean Squared Error (RMSE)'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.legend(['SAC-based 2D feedback Controller','P-ILC Controller'])
#plt.savefig('SAEforRMES.png',dpi=600)
plt.show()
pdb.set_trace()
a=2