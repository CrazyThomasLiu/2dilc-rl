import control
import numpy as np
import matplotlib.pyplot as plt
import pdb

a = 3.2
b = 0.6
c = 50
d = 0.56
k = 125
r = 1.6

def state_update(t, x, u, params):
    # Parameter setup

    # Map the states into local variable names
    z1 = x[0]
    z2 = x[1]
    # Compute the discrete updates
    dz1=0.6512*z1-0.4087*z2+0.0817*u
    dz2=0.0817*z1+0.9781*z2+0.0044*u

    return [dz1, dz2]
def ouput_update(t, x, u, params):
    # Parameter setup


    # Compute the discrete updates
    y=x[0]-x[1]

    return [y]
io_predprey = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2'),dt=0.1,name='predprey')
#pdb.set_trace()
X0 = [0.0, 0.0]                 # Initial H, L
T = np.linspace(0, 3.5,36)   # Simulation 70 years of time
#pdb.set_trace()
# Simulate the system
t, y = control.input_output_response(io_predprey, T, 1, X0)
pdb.set_trace()
# Plot the response
plt.figure(1)
plt.plot(t, y)
plt.show(block=False)
pdb.set_trace()
a=2