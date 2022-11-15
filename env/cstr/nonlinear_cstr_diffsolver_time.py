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

print("Contineous system:",control.isctime(Nonlinear_CSTR))
"1. continuous simulation"
#pdb.set_trace()
X0 = [1.0, 395.33]                 # Initial x1, x2
T = np.linspace(0, 3, 301)
#T = np.linspace(0, 0.5, 51)
#T = np.linspace(0, 2, 201)
#T = np.linspace(0, 1, 101)
#pdb.set_trace()
# Simulate the system
"RK45"
starttime=time.time()
t_con_1, y_con_1 = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='RK45')
endtime=time.time()
duringtime=endtime-starttime
print("Solver-RK45:",duringtime)
"DOP853"
starttime=time.time()
t_con_2, y_con_2 = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='DOP853')
endtime=time.time()
duringtime=endtime-starttime
print("Solver-DOP853:",duringtime)
"BDF"
starttime=time.time()
t_con_3, y_con_3 = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='BDF')
endtime=time.time()
duringtime=endtime-starttime
print("Solver-BDF:",duringtime)
"Radau"
starttime=time.time()
t_con_4, y_con_4 = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='Radau')
endtime=time.time()
duringtime=endtime-starttime
print("Solver-Radau:",duringtime)
"LSODA"
starttime=time.time()
t_con_5, y_con_5 = control.input_output_response(Nonlinear_CSTR, T, 0., X0,method='LSODA')
endtime=time.time()
duringtime=endtime-starttime
print("Solver-LOSDA:",duringtime)
# Plot the response
plt.figure(1)
plt.plot(t_con_1, y_con_1,linewidth=1)
plt.plot(t_con_2, y_con_2,linewidth=1)
plt.plot(t_con_3, y_con_3,linewidth=1)
plt.plot(t_con_4, y_con_4,linewidth=1)
plt.plot(t_con_5, y_con_5,linewidth=1)
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
plt.legend(['RK45', 'DOP853', 'BDF', 'Radau', 'LOSDA'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
ylable = 'output'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.savefig('CSTR_diffsolver_time.png',dpi=700)
plt.show(block=False)
#pdb.set_trace()
#a=2