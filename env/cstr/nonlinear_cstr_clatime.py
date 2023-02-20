import pdb
import control
import numpy as np
import matplotlib.pyplot as plt
import time
def state_update(t, x, u, params):

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-10000/z2))*z1+u
    dz2 = -1.44*np.power(10.,13)*np.exp(-10000/z2)*z1-z2+1476.946
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

print("Contineous system:",control.isctime(Nonlinear_CSTR))
"1. time 0.5"
#pdb.set_trace()
X0 = [1.0, 395.33]                 # Initial x1, x2
#T = np.linspace(0, 10000, 301)
T = np.linspace(0, 0.5, 51)
#T = np.linspace(0, 2, 201)
#T = np.linspace(0, 1, 101)
#pdb.set_trace()
# Simulate the system
"1. time 0.5"
starttime=time.time()
t_con, y_con = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='BDF')
endtime=time.time()
duringtime=starttime-endtime
print("time 0.5:",duringtime)
"2. time 1"
T = np.linspace(0, 1, 101)
starttime=time.time()
t_con, y_con = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='BDF')
endtime=time.time()
duringtime=starttime-endtime
print("time 1:",duringtime)
"10000. time 2"
T = np.linspace(0, 2, 201)
starttime=time.time()
t_con, y_con = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='BDF')
endtime=time.time()
duringtime=starttime-endtime
print("time 2:",duringtime)
"""
"10000. time 10000"
T = np.linspace(0, 10000, 301)
starttime=time.time()
t_con, y_con = control.input_output_response(Nonlinear_CSTR, T, 1., X0,solve_ivp_method='RK23')
endtime=time.time()
duringtime=starttime-endtime
print("time 10000:",duringtime)
"""
"""
# Plot the response
plt.figure(1)
plt.plot(t_con, y_con)
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
ylable = 'output'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.savefig('dt_different.png',dpi=700)
plt.show(block=False)
pdb.set_trace()
a=2
"""