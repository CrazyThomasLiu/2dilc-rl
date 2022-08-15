import control
import numpy as np
import matplotlib.pyplot as plt
import pdb
import csv
def state_update(t, x, u, params):
    # Parameter setup

    # Map the states into local variable names
    z1 = x[0]
    z2 = x[1]
    z3 = x[2]
    # Compute the discrete updates
    dz1=1.607*z1-0.6086*z2-0.9282*z3+1.239*u
    dz2=z1
    dz3 = u

    return [dz1, dz2,dz3]
def ouput_update(t, x, u, params):
    # Parameter setup


    # Compute the discrete updates
    y=x[0]

    return [y]
io_predprey = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y'),
    states=('dz1', 'dz2', 'dz3'),dt=0.1,name='predprey')
#pdb.set_trace()
X0 = [0.0, 0.0,0.0]                 # Initial H, L
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



with open('../compare/pythondiscrtesystem.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write multiple rows

    writer.writerows(map(lambda x: [x], y))
pdb.set_trace()
a=2