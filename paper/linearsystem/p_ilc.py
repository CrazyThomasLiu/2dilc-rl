import control
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import copy
import math
import csv
# define the dimensions of the state space
m=3
n=1
r=1# No useful
l=1
T_length=200
#T_length=20
batch=200

save_figure=False
save_csv=False
"1. the controlled system"
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
    y=x[0]

    return [y]
io_nonlinearsystem = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2', 'dz3'),dt=1,name='ILCsystem')
#pdb.set_trace()
"2. the reference trajectory "
y_ref=200*np.ones((T_length,1))
#pdb.set_trace()
y_ref[100:]=1.5*y_ref[100:]
"3. P-type ILC"
lmda=0.00001

X0 = np.array((0.0,0.0,0.0))
T = np.array((0.0,1))
#T=np.linspace(0, 3.5,36)
input=np.array((1.,1.))
#pdb.set_trace()

# input signal
u_k=np.zeros((1,n))
u_k_last=np.zeros((T_length,n))

# output singal
y_k_last=np.zeros((200,l))


#pdb.set_trace()
#define the output data
y_data=[]
u_data=[]

for batch_index in range(batch):
    X0 = np.array((0.0, 0.0, 0.0))
    u_data.append(copy.deepcopy(u_k_last))
    #pdb.set_trace()
    for item in range(T_length):
        #if item ==0:
        #pdb.set_trace()
        # set the continuous sample time
        T[0] = item
        T[1] = item + 1
        input[0]=u_k_last[item][0]
        input[1] =u_k_last[item][0]
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(io_nonlinearsystem, T, input, X0=X0,params={"batch_num":batch_index}, return_x=True)
        #pdb.set_trace()
        # change the initial state
        X0[0] = x_step[0][1]
        X0[1] = x_step[1][1]
        X0[2] = x_step[2][1]
        #here is important uk+1(t)=uk(t)+lmda*ek(t) 若是当前批次ek+1(t)是未来信息，不可获得
        u_k_last[item][0]=u_k_last[item][0]+lmda*(y_ref[item,0]-y_step[1])
        #pdb.set_trace()
        #change the output
        y_k_last[item,0]=y_step[1]
    y_data.append(copy.deepcopy(y_k_last))
    #pdb.set_trace()
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
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('discrete_batch_out.png',dpi=700)
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
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('discrete_batch_input.png',dpi=700)
plt.show()
#3.SAE
SAE=np.zeros(batch)
for batch_index_2d in range(batch):
    y_out_time = y_data[batch_index_2d]
    #pdb.set_trace()
    for time in range(T_length):
        SAE[batch_index_2d] += (abs(y_out_time[time] - y_ref[time])) ** 2
    SAE[batch_index_2d] = math.sqrt(SAE[batch_index_2d] / T_length)
"""save the sqrt to the csv"""
if save_csv==True:
    with open('./bathsinus.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows

        writer.writerows(map(lambda x: [x], SAE))
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
if save_figure==True:
    plt.savefig('2DILC_RMES.png',dpi=700)
plt.show()
pdb.set_trace()
a=2