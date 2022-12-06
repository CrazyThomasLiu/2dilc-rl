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

"1. define the CSTR nonlinear model"
"From the reference Chi R, Huang B, Hou Z, et al. Data‐driven high‐order terminal iterative learning control with a " \
"faster convergence speed[J]. International Journal of Robust and Nonlinear Control, 2018, 28(1): 103-119."
# define the dimensions of the state space
m=2  # dimension of the state
n=2  # input
#r=2# No useful
l=2   # output
T_length=200
#T_length=300
batch=40
batch_SAE=100
#batch_SAE=1000
save_figure=True
save_csv=True
# the equilibrium point
x_qp=np.array([[0.57336624],[395.3267527]])
u_qp=np.array([[1.],[0.]])
#pdb.set_trace()
def state_update(t, x, u, params):
    # Map the states into local variable names
    batch_num = params.get('batch_num', 0)
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1=np.array([u[0]])
    n2 = np.array([u[1]])
    # Compute the discrete updates
    a=1+0.1*np.sin(2.5*t* np.pi)+0.1*np.sin(batch_num * np.pi / 10)
    #a = 1+ 0.1 * np.sin(batch_num * np.pi / 10)
    dz1 = -(a+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+n1
    #dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - a*z2+0.041841*n2 +310*a
    #pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y1 = x[0]
    y2 = x[1]

    return [y1,y2]


Nonlinear_CSTR = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u1','u2'), outputs=('y1','y2'),
    states=('dz1', 'dz2'), dt=0, name='Nonlinear_CSTR')

print("Continuous system:",control.isctime(Nonlinear_CSTR))
"2. define the initial state "
X0 = np.array((0.47, 396.9))               # Initial x1, x2
T = np.array((0.0,0.01))
sample_time=0.01

#input=np.array((1.,1.))
#pdb.set_trace()
"3. 2D system "
#define the reference trajectory
y_ref=np.ones((T_length,2))
y_ref[:,0]=0.57*y_ref[:,0]
y_ref[:,1]=395*y_ref[:,1]
"""

y_ref=np.ones((2,T_length))
y_ref[0,:]=0.57*y_ref[0,:]
y_ref[1,:]=395*y_ref[1,:]
"""
#x_k=np.zeros((1,m))
x_k=np.array([[0.47, 396.9]])
#x_k_last=np.zeros((T_length+1,m))
#pdb.set_trace()
x_k_last=np.repeat(x_k,T_length+1,axis=0)
#x_k_current=np.zeros((T_length+1,m))
x_k_current=np.repeat(x_k,T_length+1,axis=0)
#y_k_last=np.zeros((T_length,l))
y_k_last=np.repeat(x_k,T_length,axis=0)
x_2d=np.zeros((1,l+m))

K=np.array([[-137.0651353, -50.59543044, 26.82552131, 26.77557816],
                      [11.90374147, -126.38433003, 5.65660253, 5.77072767]])
"u_k = delta_u_k + equilibrium point u"
r_k=np.zeros((n,1))
u_k=np.zeros((n,1))
u_k_last=np.zeros((T_length,n))

delta_u_k=np.zeros((n,1))
delta_u_k_last=np.zeros((T_length,n))
#pdb.set_trace()
#define the output data
y_data=[]
u_data=[]


"4. Simulation: 2d-ilc for the MIMO-CSTR system "

for batch_index in range(batch_SAE):
    x_k=np.array([[0.47, 396.9]])
    X0 = np.array((0.47, 396.9))
    #pdb.set_trace()
    for item in range(T_length):
        #if item ==0:
         #pdb.set_trace()
        # set the continuous sample time
        T[0] = sample_time*item
        T[1] = sample_time*(item + 1)
        tem_x=x_k[0]-x_k_last[item]  # 上一个批次应该是0
        #pdb.set_trace()
        tem_y=y_ref[item]-y_k_last[item]
        x_2d=np.block([[tem_x,tem_y]])
        #pdb.set_trace()
        r_k=K@x_2d.T
        delta_u_k[0, 0] = delta_u_k_last[item,0] + r_k[0, 0]
        delta_u_k[1, 0] = delta_u_k_last[item,1] + r_k[1, 0]
        u_k[0, 0] =  delta_u_k[0, 0] + u_qp[0,0]
        u_k[1, 0] =  delta_u_k[1, 0]+ u_qp[1,0]
        #pdb.set_trace()
        # constained the input

        if u_k[0,0]>10:
            u_k[0,0]=10
        elif u_k[0,0]<0:
            u_k[0,0]=0

        # input 2 is the cooling so only negative
        if u_k[1,0]<-1000:
            u_k[1,0]=-1000
        elif u_k[1,0]>1000:
            u_k[1,0]=1000
        # translate the u_k to the delta_u_k
        delta_u_k[0, 0] =  u_k[0, 0] - u_qp[0,0]
        delta_u_k[1, 0] =  u_k[1, 0]- u_qp[1,0]

        #pdb.set_trace()
        #input[0]=u_k[0,0]
        #input[1] = u_k[0,0]
        response_input=np.repeat(u_k,2,axis=1)
        #if item==147:
            #pdb.set_trace()
        #print(item)
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(Nonlinear_CSTR, T, response_input, X0=X0,params={"batch_num":batch_index}, return_x=True,method='LSODA')
        #pdb.set_trace()
        # change the initial state
        X0[0] = x_step[0,1]
        X0[1] = x_step[1,1]
        # save the data into the memory
        #u_k_last[0,item] = u_k[0, 0]
        #u_k_last[1, item] = u_k[1, 0]
        u_k_last[item,0] = u_k[0, 0]
        u_k_last[item,1] = u_k[1, 0]
        delta_u_k_last[item,0] = delta_u_k[0, 0]
        delta_u_k_last[item,1] = delta_u_k[1, 0]
        #pdb.set_trace()
        y_k_last[item]=y_step[:,1]
        #pdb.set_trace()
        for item1 in range(m):
            x_k_current[(item+1),item1]=x_step[item1,1]
            x_k[0,item1]=x_step[item1,1]   #change the current information
        #x_k_last[item]=x_step[1]
        #print(y_step[1])
        #pdb.set_trace()
    x_k_last=copy.deepcopy(x_k_current)
    #pdb.set_trace()
    y_data.append(copy.deepcopy(y_k_last))
    u_data.append(copy.deepcopy(u_k_last))
#pdb.set_trace()


# Plot the 3d visibale figure
"1.Outpur Response"
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


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Output:Production Concentration $kmol/m^{3}$'
zlable = '$y_{1}$:Production Concentration $kmol/m^{3}$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
ax.legend(['y_Ref','y_out'])
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('constrain_1000_n1000_30%_y1.png',dpi=700)
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
ax.legend(['y_Ref','y_out'])
ax.view_init(40, -19)
if save_figure==True:
    plt.savefig('constrain_1000_n1000_30%_y2.png',dpi=700)
#plt.show()
#pdb.set_trace()
"2.Input Signal"
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }

"2.1 u1 Feed Concentration"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_data[item2][:,0].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Feed Concentration $kmol/m^{3}$'
zlable = '$u{1}$:Feed Concentration $kmol/m^{3}$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
#ax.legend(['y_Ref','y_out'])
#ax.view_init(52, -16)
ax.view_init(40, -19)
#ax.view_init(0, 90)
if save_figure==True:
    plt.savefig('constrain_1000_n1000_30%_u1.png',dpi=700)
#plt.show()
#pdb.set_trace()

"2.2 u2 Heat"
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(T_length)
Batch_length=np.ones(T_length,dtype=int)
for item2 in range(batch):
    batch_plot=Batch_length*(item2+1)
    if (item2%2)==0:
        ax.plot3D(batch_plot,t, u_data[item2][:,1].squeeze(),linewidth=1,color='black')


xlable = 'Batch:k'
ylable = 'Time:t'
#zlable = 'Input:Heat $kJ/min$'
zlable = '$u_{2}$:Heat $kJ/min$'
ax.set_xlabel(xlable,font2)
ax.set_ylabel(ylable,font2)
ax.set_zlabel(zlable,font2)
#ax.legend(['y_Ref','y_out'])
#ax.view_init(52, -16)
ax.view_init(40, -19)
#ax.view_init(0, 90)
if save_figure==True:
    plt.savefig('constrain_1000_n1000_30%_u2.png',dpi=700)
plt.show()
#pdb.set_trace()

"3.SAE"
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
"3.1 Calculation of the SAE"
SAE_y1=np.zeros(batch_SAE)
SAE_y2=np.zeros(batch_SAE)
#cal the sum of the y1 and y2
SAE_sum=np.zeros(batch_SAE)
for batch_index in range(batch_SAE):
    y_list_y1 = y_data[batch_index][:,0]
    y_list_y2 = y_data[batch_index][:, 1]
    #pdb.set_trace()
    for time in range(T_length):
        SAE_y1[batch_index] += abs(y_list_y1[time] - y_ref[time,0])
        SAE_y2[batch_index] += abs(y_list_y2[time] - y_ref[time, 1])
    #SAE_y1[batch_index] = math.sqrt(SAE_y1[batch_index] / T_length)
    #SAE_y2[batch_index] = math.sqrt(SAE_y2[batch_index] / T_length)
    SAE_sum[batch_index]=SAE_y1[batch_index]+SAE_y2[batch_index]
#plt.figure()
#pdb.set_trace()
if save_csv==True:
    """save the sqrt to the csv"""
    with open('./constrain_1000_n1000_30%_SAE_y1.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows

        writer.writerows(map(lambda x: [x], SAE_y1))

    with open('./constrain_1000_n1000_30%_SAE_y2.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows

        writer.writerows(map(lambda x: [x], SAE_y2))
    with open('./constrain_1000_n1000_30%_SAE_sum.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows

        writer.writerows(map(lambda x: [x], SAE_sum))
"3.2 Plot of the y1"
plt.subplot(2,1,1)
batch_time=range(1,batch_SAE+1)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_y1,linewidth=3,color='blue',linestyle=':')
#plt.plot(batch_time,SAE_tem,linewidth=3,color='red',linestyle=':')
plt.grid()
xlable = 'Batch:k'
#ylable = 'Root Mean Squared Error (RMSE)'
ylable = 'RMSE'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.legend(['Production Concentration','Temperature'])
plt.legend(['Production Concentration'])

"3.3 Plot of the y2"

plt.subplot(2,1,2)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_y2,linewidth=3,color='red',linestyle=':')
plt.grid()
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['Temperature'])
if save_figure==True:
    plt.savefig('constrain_1000_n1000_30%_SAE.png',dpi=700)


"3.3 Plot of the sum SAE"

plt.figure()
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,SAE_sum,linewidth=3,color='red',linestyle=':')
plt.grid()
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['The Total SAE'])
if save_figure==True:
    plt.savefig('constrain_1000_n1000_total_30%_SAE.png',dpi=700)

plt.show()
pdb.set_trace()
a=2