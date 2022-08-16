import control
import numpy as np
import matplotlib.pyplot as plt
import pdb
#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
import csv
def state_update(t, x, u, params):
    # Parameter setup

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3 = np.array([x[2]])
    # Compute the discrete updates
    dz1=1.607*z1-0.6086*z2-0.9282*z3+1.239*u
    dz2=z1
    dz3 = u
    #pdb.set_trace()
    return [dz1, dz2,dz3]
def ouput_update(t, x, u, params):
    # Parameter setup


    # Compute the discrete updates
    y=x[0]

    return [y]
io_nonlinearsystem = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2', 'dz3'),dt=0.1,name='ILCsystem')
#pdb.set_trace()
#X0 = [0.0, 0.0,0.0]
X0 = np.array((0.,0.,0.))
T = np.array((0.0,0.1))
input=np.array((1.,1.))
#pdb.set_trace()
# Simulate the system
#t, y,x = control.input_output_response(io_nonlinearsystem, T, input, X0,return_x=True)
#t, y = control.input_output_response(io_nonlinearsystem, T, input, X0)
y=np.zeros(301)
#pdb.set_trace()
for item in range(300):
    t_step, y_step, x_step = control.input_output_response(io_nonlinearsystem, T, input, X0=X0, return_x=True)
    #pdb.set_trace()
    #T[0]=T[1]
    #T[1]=T[1]+0.1
    #pdb.set_trace()
    X0[0] = x_step[0][1]
    X0[1] = x_step[1][1]
    X0[2] = x_step[2][1]
    #pdb.set_trace()
    y[item+1]=y_step[1]
pdb.set_trace()
# Plot the response
plt.figure(1)
plt.plot(t, y)
plt.show(block=False)
pdb.set_trace()



pdb.set_trace()
a=2