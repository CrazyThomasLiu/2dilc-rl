import pdb
import control
import numpy as np
import matplotlib.pyplot as plt
"Test: How to set the multi-input "
"From the reference Chi R, Huang B, Hou Z, et al. Data‐driven high‐order terminal iterative learning control with a " \
"faster convergence speed[J]. International Journal of Robust and Nonlinear Control, 2018, 28(1): 103-119."
def state_update(t, x, u, params):

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1=np.array([u[0]])
    n2 = np.array([u[1]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+n1
    #dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 50000) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2+0.041841*n2 +310
    # pdb.set_trace()
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

print("Contineous system:",control.isctime(Nonlinear_CSTR))
"1. continuous simulation "
#pdb.set_trace()
X0 = [0.47, 396.9]                 # Initial x1, x2
T = np.linspace(0, 3, 301)
#T = np.linspace(0, 0.5, 51)
#T = np.linspace(0, 2, 201)
#T = np.linspace(0, 1, 101)
#pdb.set_trace()
# Simulate the system
U=np.zeros((2,1))
U[0][0]=1.0

t_con, y_con,x_con = control.input_output_response(Nonlinear_CSTR, T, [1.0,0.0], X0,return_x=True,method='LSODA')

#pdb.set_trace()

"2. continuous simulation with  different sample time "
X0 = [0.47, 396.9]                 # Initial x1, x2
T = np.linspace(0, 3, 301)

U=np.zeros((2,301))
U[0]=1.0
#pdb.set_trace()
#U=np.ones((2,301))
# Simulate the system
t_con_diff, y_con_diff,x_con_diff = control.input_output_response(Nonlinear_CSTR, T, U, X0,return_x=True,method='LSODA')

#pdb.set_trace()

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
"Simulation 1"
ax1.plot(t_con, x_con[0],color='b')
ax2.plot(t_con, x_con[1],color='r')

"Simulation 2"
ax1.plot(t_con_diff, x_con_diff[0],linestyle='dashed')
ax2.plot(t_con_diff, x_con_diff[1],linestyle='dashed')
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
y1lable = 'Production Concentration'
y2lable = 'temperature/K'
ax1.set_xlabel(xlable,font2 )
ax1.set_ylabel(y1lable,font2,color='b' )
ax2.set_ylabel(y2lable,font2,color='r' )
plt.savefig('CSTR_MIMO.png',dpi=700)
plt.show(block=False)

pdb.set_trace()
a=2