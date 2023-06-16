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
batch=50
save_figure=False
save_csv=True
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
X0 = np.array((0.0,0.0,0.0))
T = np.array((0.0,1))
input=np.array((1.,1.))

#define the 2D systems
y_ref=200*np.ones((T_length,1))
#pdb.set_trace()
y_ref[100:]=1.5*y_ref[100:]
x_k=np.zeros((1,m))
x_k_last=np.zeros((200+1,m))
x_k_current=np.zeros((200+1,m))
y_k_last=np.zeros((200,l))
x_2d=np.zeros((1,l+m))
K=np.array([[-1.4083788,0.57543156,0.87756631,0.71898388]])
#K=np.array([[-1.4201,0.58403,0.89073,0.70219]])
r_k=np.zeros((1,n))
u_k=np.zeros((1,n))
u_k_last=np.zeros((T_length,n))
#define the output data
y_data=[]
u_data=[]

u_ilc_time_transction=[]
# Simulate the system
for batch_index in range(batch):
    x_k=np.zeros((1,m))
    X0 = np.array((0.0, 0.0, 0.0))
    for item in range(T_length):
        # set the continuous sample time
        T[0] = item
        T[1] = item + 1
        tem_x=x_k[0]-x_k_last[item]
        tem_y=y_ref[item]-y_k_last[item]
        x_2d=np.block([[tem_x,tem_y]])
        r_k[0][0]=K@x_2d.T
        u_k[0][0]=u_k_last[item][0]+r_k[0][0]
        input[0]=u_k[0][0]
        input[1] = u_k[0][0]
        t_step, y_step, x_step = control.input_output_response(io_nonlinearsystem, T, input, X0=X0,params={"batch_num":batch_index}, return_x=True)
        # change the initial state
        X0[0] = x_step[0][1]
        X0[1] = x_step[1][1]
        X0[2] = x_step[2][1]
        # save the data into the memory
        u_k_last[item][0]=u_k[0][0]
        y_k_last[item]=y_step[1]
        if item == 50:
            u_ilc_time_transction.append(u_k[0][0])
        for item1 in range(m):
            x_k_current[(item+1)][item1]=x_step[item1][1]
            x_k[0][item1]=x_step[item1][1]   #change the current information

    x_k_last=copy.deepcopy(x_k_current)
    #deep copy
    y_data.append(copy.deepcopy(y_k_last))
    u_data.append(copy.deepcopy(u_k_last))

#SAE
SAE=np.zeros(batch)
for batch_index_2d in range(batch):
    y_out_time = y_data[batch_index_2d]
    #pdb.set_trace()
    for time in range(T_length):
        SAE[batch_index_2d] += (abs(y_out_time[time] - y_ref[time])) ** 2
    SAE[batch_index_2d] = math.sqrt(SAE[batch_index_2d] / T_length)
"""save the sqrt to the csv"""
if save_csv==True:
    with open('./2DILC_RMSE_injection_molding_process.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows
        writer.writerow(['Value'])
        writer.writerows(map(lambda x: [x], SAE))
plt.show()
