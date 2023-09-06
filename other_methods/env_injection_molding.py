import sys
import os
import pdb
import control
import numpy as np

class  BatchSysEnv:
    def __init__(self):
        'set the hyperparameters'
        #define the dim of the state space
        self.m = 3
        self.n = 1
        self.r = 1  # No useful None
        self.l = 1
        # sample frequency
        self.T=1.0
        # initial sample time
        self.T_in=np.array((0.0,0.0))
        # initial state
        self.X0=np.array((0.0, 0.0, 0.0))
        # duration steps per batch
        self.T_length=200
        # batch counter
        self.batch_num = 0
        # time counter
        self.time=0
        # define the reference trajectory
        self.y_ref = 200 * np.ones((self.T_length, 1))
        self.y_ref[100:] = 1.5 * self.y_ref[100:]
        'define the batch system'
        def state_update(t, x, u, params):
            # get the parameter from the params
            batch_num = params.get('batch_num', 0)
            # Parameter setup
            # non-repetitive nature
            sigma1 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            sigma2 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            sigma3 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            sigma4 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            # model mismatch
            sigma1 += 0.5 * np.sin(0.1 * t)
            sigma2 += 0.5 * np.sin(0.1 * t)
            sigma3 += 0.5 * np.sin(0.1 * t)
            sigma4 += 0.5 * np.sin(0.1 * t)
            # Map the states into local variable names
            z1 = np.array([x[0]])
            z2 = np.array([x[1]])
            z3 = np.array([x[2]])
            # Compute the discrete updates
            dz1 = (1.607 + 0.0804 * sigma1) * z1 - (0.6086 + 0.0304 * sigma2) * z2 - (0.9282 + 0.0464 * sigma3) * z3 + (
                    1.239 + 0.062 * sigma4) * u
            dz2 = z1
            dz3 = u
            # pdb.set_trace()
            return [dz1, dz2, dz3]

        def ouput_update(t, x, u, params):
            # Parameter setup

            # Compute the discrete updates
            y = x[0]

            return [y]

        self.sys = control.NonlinearIOSystem(
            state_update, ouput_update, inputs=('u'), outputs=('y'),
            states=('dz1', 'dz2', 'dz3'), dt=1, name='injection_modeling')

        self.input_signal = np.zeros((self.n,1))

    def reset(self):
        # reset the initial sample time
        self.T_in = np.array((0.0, 0.0))
        # reset the initial state
        self.X0=np.array((0.0, 0.0, 0.0))
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
        self.X0[2] = x_step[2][1]

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
    T_length = 200
    # initial the controlled system
    controlled_system = BatchSysEnv()
    for batch_index in range(batch):
        out = controlled_system.reset()
        pdb.set_trace()
        for item in range(T_length):
            control_signal=1.0
            out = controlled_system.step(control_signal)
            #pdb.set_trace()
    pdb.set_trace()
    a=2