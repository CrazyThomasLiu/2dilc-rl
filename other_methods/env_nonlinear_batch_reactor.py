import sys
import os
import pdb
import control
import numpy as np

class  BatchSysEnv:
    def __init__(self):
        'set the hyperparameters'
        #define the dim of the state space
        self.m = 2
        self.n = 1
        self.r = 1  # No useful None
        self.l = 1
        # sample frequency
        self.T=0.01
        # initial sample time
        self.T_in=np.array((0.0,0.0))
        # initial state
        self.X0=np.array((0.5, 310.))
        # duration steps per batch
        self.T_length=300
        # batch counter
        self.batch_num = 0
        # time counter
        self.time=0
        # define the reference trajectory
        self.y_ref = np.ones((self.T_length, 1))
        self.y_ref[0:100, 0] = 370 * self.y_ref[0:100, 0]
        self.y_ref[100:T_length, 0] = 350 * self.y_ref[100:T_length, 0]
        'define the batch system'

        def state_update(t, x, u, params):
            batch_num = params.get('batch_num', 0)
            # Compute the discrete updates
            a = 1 + 1 * np.sin(2.5 * t * np.pi) + 1 * np.sin(batch_num * np.pi / 10)
            # Map the states into local variable names
            z1 = np.array([x[0]])
            z2 = np.array([x[1]])
            n1 = np.array([u[0]])
            # Compute the discrete updates
            dz1 = -(1 + 7.2 * np.power(10., 10) * np.exp(-np.power(10., 4) / z2)) * z1 + 0.5
            dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 0.041841 * n1 + 310 * a
            return [dz1, dz2]

        def ouput_update(t, x, u, params):
            # Parameter setup

            # Compute the discrete updates
            y1 = x[1]

            return [y1]

        self.sys = control.NonlinearIOSystem(
            state_update, ouput_update, inputs=('u1'), outputs=('y1'),
            states=('dz1', 'dz2'), dt=0, name='SISO_CSTR')

        self.input_signal = np.zeros((self.n,1))

    def reset(self):
        # reset the initial sample time
        self.T_in = np.array((0.0, 0.0))
        # reset the initial state
        self.X0=np.array((0.5, 310.))
        # reset the time
        self.time = 0

        return self.X0


    def step(self,action):
        # the control input
        self.input_signal[0,0]=action
        response_input = np.repeat(self.input_signal, 2, axis=1)
        # simulation time
        #pdb.set_trace()
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T

        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, response_input, X0=self.X0,params={"batch_num":self.batch_num}, return_x=True)
        # the state space
        # change the initial state
        self.X0[0] = x_step[0][1]
        self.X0[1] = x_step[1][1]

        # the current time
        self.time+=1
        if self.time==self.T_length:
            self.batch_num += 1  # +1  ???
            done=1
        else:
            done=0

        #conver the np.array to the list
        state=self.X0.tolist()
        #pdb.set_trace()
        return state



if __name__=="__main__":
    # set the simulation batches
    batch = 10
    T_length = 300
    # initial the controlled system
    controlled_system = BatchSysEnv()
    for batch_index in range(batch):
        out = controlled_system.reset()
        #pdb.set_trace()
        for item in range(T_length):
            control_signal=1.0
            out = controlled_system.step(control_signal)
            pdb.set_trace()
    pdb.set_trace()
    a=2