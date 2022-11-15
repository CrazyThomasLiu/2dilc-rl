import pdb
import control
import numpy as np
import matplotlib.pyplot as plt

"From the reference Chi R, Hou Z, Jin S, et al. Computationally efficient data-driven higher order optimal iterative " \
"learning control[J]. IEEE transactions on neural networks and learning systems, 2018, 29(12): 5971-5980."
def state_update(t, x, u, params):

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+u
    #dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 1476.946
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
t_con, y_con,x_con = control.input_output_response(Nonlinear_CSTR, T, 1., X0,return_x=True,method='LSODA')

#pdb.set_trace()
"""
plt.figure(1)
plt.plot(t_con, y_con)
plt.plot(t_con, x_con[1])
#plt.legend(['Continous', 'Discrete T=1', 'Discrete T=0.5', 'Discrete T=0.1'])
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
xlable = 'time '
ylable = 'output'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.savefig('CSTR.png',dpi=700)
plt.show(block=False)
"""
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t_con, y_con,color='b')
ax2.plot(t_con, x_con[1],color='r')
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
#plt.savefig('CSTR.png',dpi=700)
plt.show(block=False)

pdb.set_trace()
a=2