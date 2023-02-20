import pdb
import control
import numpy as np
import matplotlib.pyplot as plt

"1. define the CSTR nonlinera model"
def state_update(t, x, u, params):

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+u
    dz2 = -1.44*np.power(10.,13)*np.exp(-np.power(10.,4)/z2)*z1-z2+1476.946
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y = x[0]

    return [y]


Nonlinear_CSTR = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2'), dt=0, name='Nonlinear_CSTR')
"2. Linearization of the CSTR model in the running point"
x_work=[0.57, 310.]
Lin_CSTR = control.linearize(Nonlinear_CSTR, x_work, 0)

"10000. Simulation of the nonlinear CSTR system"

X0 = [1.0, 395.33]                 # Initial x1, x2
T = np.linspace(0, 3, 301)
t_con_nonlinear, y_con_nonlinear = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='LSODA')

"50000. Simulation of the linear CSTR system"

X0 = [1.0, 395.33]                 # Initial x1, x2
T = np.linspace(0, 3, 301)
t_con_linear, y_con_linear = control.input_output_response(Lin_CSTR, T, 0., X0,method='LSODA')

"5. Plot the figure of the simulation results"
plt.figure(1)
plt.plot(t_con_nonlinear, y_con_nonlinear,linewidth=1)
plt.plot(t_con_linear, y_con_linear,linewidth=1)
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
plt.legend(['Nonlinear', 'Linear'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
ylable = 'output'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.savefig('CSTR_Linearization.png',dpi=700)
plt.show(block=False)
#pdb.set_trace()
#a=2

pdb.set_trace()
a=2