import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import control
import copy
import typing
import pprint

class BatchSysEnv:
    def __init__(self,T_length,sys,ak=-10,T=1.,X0=np.array((0.0, 0.0, 0.0))):
        self.sys=sys
        self.T=T
        self.T_in=np.array((0.0,0.0))
        self.X0=X0
        self.T_length=T_length
        self.env_num=1
        self.batch_num = 0
        self.time=0
        # define the reference trajectory
        self.y_ref = 200 * np.ones((self.T_length, 1))
        self.y_ref[100:] = 1.5 * self.y_ref[100:]
        #pdb.set_trace()
        #define the dim of the state space
        self.m = 3
        self.n = 1
        self.r = 1  # No useful
        self.l = 1
        # define the 2DILC Controller
        self.K = np.array([[-1.4083788, 0.57543156, 0.87756631, 0.71898388]])
        self.x_k = np.zeros((1, self.m))
        self.x_k_last = np.zeros((200 + 1, self.m))
        self.x_k_current = np.zeros((200 + 1, self.m))
        self.y_k_last = np.zeros((200, self.l))
        # Wsigma_k=np.zeros((1,m))
        # sigma_k=x_k[0]-x_k_last[0]
        # e_k=np.zeros((1,l))
        # merge the sigma_k and e_k
        self.x_2d = np.zeros((1, self.l + self.m))
        # K=np.array([[-1.4201,0.58403,0.89073,0.70219]])
        self.r_k = np.zeros((1, self.n))
        self.u_k = np.zeros((1, self.n))
        self.u_k_last = np.zeros((self.T_length, self.n))
        self.input_signal = np.array((0., 0.))






    def reset(self):
        # set the sample time
        #pdb.set_trace()
        self.T_in = np.array((0.0, 0.0))
        self.X0=np.array((0.0, 0.0, 0.0))
        self.time = 0
        self.x_k = np.zeros((1, self.m))
        self.input_signal = np.array((0., 0.))
        self.x_k_last = copy.deepcopy(self.x_k_current)
        self.batch_num+=1 # +1
        pass




    def step(self,action):
        # set the continuous sample time
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T
        # cal the 2D system state matrix
        #pdb.set_trace()
        tem_x=self.x_k[0]-self.x_k_last[self.time]  # 上一个批次应该是0
        tem_y=self.y_ref[self.time]-self.y_k_last[self.time]
        self.x_2d=np.block([[tem_x,tem_y]])
        #pdb.set_trace()
        #cal the control signal
        self.r_k[0]=self.K@self.x_2d.T
        self.u_k[0]= self.u_k_last[self.time]+ self.r_k[0]
        self.input_signal[0]=self.u_k[0][0]
        self.input_signal[1] =self.u_k[0][0]
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, self.input_signal, X0=self.X0,params={"batch_num":self.batch_num}, return_x=True)
        #pdb.set_trace()
        # change the initial state
        self.X0[0] = x_step[0][1]
        self.X0[1] = x_step[1][1]
        self.X0[2] = x_step[2][1]
        # save the data into the memory
        self.u_k_last[self.time]=self.u_k[0]
        self.y_k_last[self.time]=y_step[1]
        for item1 in range(self.m):
            self.x_k_current[(self.time+1)][item1]=x_step[item1][1]
            self.x_k[0][item1]=x_step[item1][1]   #change the current information
        # the current time
        self.time+=1

        return y_step[1]

    def close(self):
        pass








if __name__=="__main__":
    # set the hyperparameters
    T_length = 200
    X0 = np.array((0.0, 0.0, 0.0))
    #T = np.array((0.0, 1))
    # define the batch system
    def state_update(t, x, u, params):
        # Parameter setup

        # Map the states into local variable names
        z1 = np.array([x[0]])
        z2 = np.array([x[1]])
        z3 = np.array([x[2]])
        # Compute the discrete updates
        dz1 = 1.607 * z1 - 0.6086 * z2 - 0.9282 * z3 + 1.239 * u
        dz2 = z1
        dz3 = u
        # pdb.set_trace()
        return [dz1, dz2, dz3]
    """
    def state_update(t, x, u, params):
        # get the parameter from the params
        batch_num = params.get('batch_num', 0)
        # Parameter setup
        # pdb.set_trace()
        sigma1 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
        sigma2 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
        sigma3 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
        sigma4 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)

        # pdb.set_trace()
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
    """

    def ouput_update(t, x, u, params):
        # Parameter setup

        # Compute the discrete updates
        y = x[0]

        return [y]


    batch_system = control.NonlinearIOSystem(
        state_update, ouput_update, inputs=('u'), outputs=('y'),
        states=('dz1', 'dz2', 'dz3'), dt=1, name='ILCsystem')
    controlled_system=BatchSysEnv(T_length=T_length,sys=batch_system,X0=X0)
    out=[]
    out_batch=[]
    for batch in range(20):
        for item in range(T_length):
            out_tem=controlled_system.step(1.)
            out.append(out_tem)
        out_batch.append(out)
        controlled_system.reset()
        out = []



    pdb.set_trace()
    #############################################################
    #show
    # Plot the response
    plt.figure()
    xlable = 't/s'
    ylable = 'Amplitude'
    title = 'Nonlinear System in Python and Matlab'
    plt.plot(T, Y_step, '--', color='red')
    plt.plot(T, Y_env)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.grid()
    plt.title(title)
    plt.legend(['Python', 'sys'], loc=1)
    # plt.savefig('Compare.png')
    plt.show()
    #######################################################
    pdb.set_trace()
    a=2