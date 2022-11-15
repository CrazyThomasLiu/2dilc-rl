import pdb
import control
import numpy as np
import matplotlib.pyplot as plt
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

X0 = [1.0, 395.33]                 # Initial x1, x2

eqpt = control.find_eqpt(Nonlinear_CSTR, X0, 0.)

pdb.set_trace()
a=2