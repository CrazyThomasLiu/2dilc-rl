import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
import pdb
import numpy as np
import control
import typing
import pprint

def updatestep(t, x, u, params):
    # Parameter setup

    #Map the states into local variable names
    x1=x[0]
    x2 = x[1]
    x3 = x[2]

    dx1 = 1.607*x1-0.6086*x2-0.9282*x3+1.239*u
    dx2 = x1
    dx3 = u


    return [dx1,dx2,dx3]

sys = control.NonlinearIOSystem(
    updatestep, None, inputs=('u'), outputs=('x1','x2','x3'),
    states=('x1','x2','x3'), name='nonlinear')


X0 = [0, 0,0]                 # Initial H, L
T = np.linspace(0, 70, 500)   # Simulation 70 years of time

# Simulate the system
t, y = control.input_output_response(sys, T, 0, X0)

# Plot the response
plt.figure(1)
plt.plot(t, y[0])
plt.plot(t, y[1])
plt.legend(['Hare', 'Lynx'])
plt.show(block=False)