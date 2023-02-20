import pdb
import control
import numpy as np
import matplotlib.pyplot as plt


def state_update(t, x, u, params):

    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    n1=np.array([u[0]])
    # Compute the discrete updates
    dz1 = -(1+7.2*np.power(10.,10)*np.exp(-np.power(10.,4)/z2))*z1+0.5
    #dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 50000) / z2) * z1 - z2 + 1476.946
    dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2+0.041841*n1 +310
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup

    # Compute the discrete updates
    y1 = x[1]

    return [y1]


Nonlinear_CSTR = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u1'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=0, name='SISO_CSTR')

print("Contineous system:",control.isctime(Nonlinear_CSTR))
"2. Linearization of the CSTR model in the running point"
x_eqt=[0.48632751, 350.]
u_eqt=[890.6456]
#u_eqt=[0.0, 0.0]
#x_eqt=[0.47, 396.9]
#u_eqt=[0.08566041727, 0.02289420772]
Lin_CSTR= control.linearize(Nonlinear_CSTR, x_eqt, u_eqt)
print("Contineous system:",control.isctime(Lin_CSTR))
#pdb.set_trace()

pdb.set_trace()
a=2