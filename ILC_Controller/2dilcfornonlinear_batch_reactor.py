import control
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import copy
import math
import csv

"1. define the CSTR nonlinear model"
"From the reference Chi R, Huang B, Hou Z, et al. Data‐driven high‐order terminal iterative learning control with a " \
"faster convergence speed[J]. International Journal of Robust and Nonlinear Control, 2018, 28(1): 103-119."
# define the dimensions of the state space
m=2  # dimension of the state
n=1  # input
#r=2# No useful
l=1   # output
T_length=300
batch=50
save_figure=False
save_csv=True
# the equilibrium point
x_qp=np.array([[0.48632751],[350.]])
u_qp=np.array([[890.6456]])
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


Nonlinear_CSTR = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u1'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=0, name='SISO_CSTR')

"2. define the initial state "
X0 = np.array((0.5, 310.))               # Initial x1, x2
T = np.array((0.0,0.01))
sample_time=0.01


"1. 2D system "
#define the reference trajectory
y_ref=np.ones((T_length,1))
y_ref[0:100,0]=370*y_ref[0:100,0]
y_ref[100:T_length,0]=350*y_ref[100:T_length,0]

x_k=np.array([[0.5, 310.]])
x_k_last=np.repeat(x_k,T_length+1,axis=0)
x_k_current=np.repeat(x_k,T_length+1,axis=0)
y_k_last=np.repeat([[310.]],T_length,axis=0)
x_2d=np.zeros((1,l+m))
K=np.array([[1185.5239983, -2045.16033935,1017.79150778]])

"u_k = delta_u_k + equilibrium point u"
r_k=np.zeros((n,1))
u_k=np.zeros((n,1))
u_k_last=np.zeros((T_length,n))

delta_u_k=np.zeros((n,1))
delta_u_k_last=np.zeros((T_length,n))
y_data=[]
u_data=[]


"2. Simulation: 2d-ilc for the nonlinear batch reactor system "

for batch_index in range(batch):
    x_k=np.array([[0.5, 310.]])
    X0 = np.array((0.5, 310.))
    for item in range(T_length):
        # set the continuous sample time
        T[0] = sample_time*item
        T[1] = sample_time*(item + 1)
        tem_x=x_k[0]-x_k_last[item]
        tem_y=y_ref[item]-y_k_last[item]
        x_2d=np.block([[tem_x,tem_y]])
        r_k=K@x_2d.T
        delta_u_k[0, 0] = delta_u_k_last[item,0] + r_k[0, 0]
        u_k[0, 0] =  delta_u_k[0, 0] + u_qp[0,0]
        # translate the u_k to the delta_u_k
        delta_u_k[0, 0] =  u_k[0, 0] - u_qp[0,0]

        response_input=np.repeat(u_k,2,axis=1)

        t_step, y_step, x_step = control.input_output_response(Nonlinear_CSTR, T, response_input, X0=X0,params={"batch_num":batch_index}, return_x=True,method='LSODA')
        # change the initial state
        X0[0] = x_step[0,1]
        X0[1] = x_step[1,1]
        # save the data into the memory
        u_k_last[item,0] = u_k[0, 0]
        delta_u_k_last[item,0] = delta_u_k[0, 0]
        y_k_last[item]=y_step[1]

        for item1 in range(m):
            x_k_current[(item+1),item1]=x_step[item1,1]
            x_k[0,item1]=x_step[item1,1]   #change the current information

    x_k_last=copy.deepcopy(x_k_current)
    y_data.append(copy.deepcopy(y_k_last))
    u_data.append(copy.deepcopy(u_k_last))

batch_time=range(1,batch+1)
"Calculation of the RMSE"
RMSE_y1=np.zeros(batch)
for batch_index in range(batch):
    y_list_y1 = y_data[batch_index][:,0]
    for time in range(T_length):
        RMSE_y1[batch_index] += (abs(y_list_y1[time] - y_ref[time,0])) ** 2
    RMSE_y1[batch_index] = math.sqrt(RMSE_y1[batch_index] / T_length)

"""save the sqrt to the csv"""
if save_csv==True:
    with open('2DILC_RMSE_nonlinear_batch_reactor.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Value'])
        writer.writerows(map(lambda x: [x], RMSE_y1))
