import pdb
import control
import numpy as np
import matplotlib.pyplot as plt

"1. define the CSTR nonlinear model"
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
"2. Linearization of the CSTR model in the running point"

x_eqt=[0.57336624, 395.3267527]
u_eqt=[1.0, 0.0]
Lin_CSTR= control.linearize(Nonlinear_CSTR, x_eqt, u_eqt)

"10000. Discretization of the linear CSTR model in sample time 0.01 "
Dis_Lin_CSTR_statespace=control.c2d(Lin_CSTR,0.01,method='zoh')
Dis_Lin_CSTR=control.LinearIOSystem(Dis_Lin_CSTR_statespace,dt=0.01)
#pdb.set_trace()
"50000. Simulation of the linear CSTR system"

X0 = [-0.10336624, 1.5732473]                  # Initial x1, x2
T = np.linspace(0, 2, 201)
t_con_linear, y_con_linear = control.input_output_response(Lin_CSTR, T, 0., X0,method='LSODA')

"5. Simulation of the discrete linear CSTR system"

X0 = [-0.10336624, 1.5732473]                  # Initial x1, x2
T = np.linspace(0, 2, 201)
t_con_dis, y_con_dis = control.input_output_response(Dis_Lin_CSTR, T, 0., X0,method='LSODA')


"6. Plot the figure of the simulation results"
"""the first figure"""
plt.subplot(2,1,1)
plt.plot(t_con_linear, y_con_linear[0],linewidth=1)
plt.plot(t_con_dis, y_con_dis[0],linestyle='dotted',linewidth=1)
#plt.plot(t_con_linear, y_con_linear[0],linewidth=1)
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
plt.legend(['Linear', 'Discrete'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
ylable = 'Production Concentration'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2,color='b' )
"""the second figure"""
plt.subplot(2,1,2)
plt.plot(t_con_linear, y_con_linear[1],linewidth=1)
plt.plot(t_con_dis, y_con_dis[1],linestyle='dotted',linewidth=1)
plt.legend(['Linear', 'Discrete'])
xlable = 'time '
ylable = 'Temperature/K'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2,color='r' )
plt.savefig('CSTR_MIMO_Discretization.png',dpi=700)
plt.show(block=False)
#pdb.set_trace()
#a=2

pdb.set_trace()
a=2