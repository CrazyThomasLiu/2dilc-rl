import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import control
import typing
import pprint

#SISO first
"""
parameters:
sys  the controlled system
N    the repeat length
T    the sample time
X0   the initial state space



"""

class NonlinearSysEnv:
    def __init__(self,N,sys,ak=-10,T=1,X0=[0.],):
        self.sys=sys
        self.T=T
        self.T_in=np.zeros(2)
        self.X0=np.array(X0)
        self.N=N
        self.env_num=1
        self.batch_num = 0
        """
        weight
        sinus

        """
        #self.er_lim=0.0001
        self.er_lim = 0.00001
        #第一个公式
        self.constant=0.5
        #self.ac_co_1=-10
        #self.ak1=-50
        self.ak1 = -4
        self.au_1 = -0.1
        #self.ac_co_l_1 = -0.1
        self.au_l_1 = -0.1
        self.er_l_1=-0.5


        #第二个公式
        self.ak=-4 ###weight
        self.au_2=-0.1
        #self.ac_co_l_2 = -0.1
        self.au_l_2 = -0.1
        self.er_l_2=-0.5
        #self.er_co_l_2 =0
        #self.scale=0.5
        self.scale = 0.8
        ###undefined varaible


        ###undefined varaible
        """
        self.y_ref = np.zeros(N)
        for item in range(N):
            self.y_ref[item] = (np.sin((item) * 2 * np.pi / N)) ** 3

        self.y_ref = np.zeros(N)
        for item in range(N):
            self.y_ref[item] = (np.sin((item) * 2 * np.pi / N))
        """
        ### set the reference trajactory
        self.y_ref = np.zeros(N)
        sum_N=0
        for item in range(int(N/3)):
            self.y_ref[item] = 0
            self.y_ref[item+60] = 3
            self.y_ref[item+120] = 1
        #reset()
        #pdb.set_trace()
        self.x = self.X0
        self.N_counter=1
        # environment information
        self.env_name="Nonlinearsys_time_batch"
        #self.state_dim=3
        self.state_dim=12
        self.action_dim=1
        self.max_step=N
        self.if_discrete=False
        self.target_return = 3.5
        self.episode_return = 0.0
        self.action=np.zeros(2)
        #self.y_re=1
        ###################
        #new state space
        self.e_t=0.
        self.e_t_1 = 0.
        self.e_t_2 = 0.
        self.u_t=0.
        self.u_t_1 = 0.
        self.u_t_2 = 0.
        self.y_ref_t=0.0
        self.y_ref_t_1 = 0.0
        self.y_ref_t_future = self.y_ref[0]
        # last batch information
        self.e_l = list(np.zeros(N))   # last batch error
        self.u_l =list(np.zeros(N))    # last batch action
        self.y_ref_l=list(np.zeros(N)) # last batch reference
        self.y_out_l=list(np.zeros(N)) # last batch out
        self.zero=0.0
        self.y_out_t=0.0
        self.y_out_t_1 = 0.0







    def reset(self):
        self.T_in=np.zeros(2)
        self.x=self.X0
        self.N_counter = 1
        self.e_t=0.
        self.e_t_1 = 0.
        self.e_t_2 = 0.
        self.u_t=0.
        self.u_t_1 = 0.
        self.u_t_2 = 0.
        self.y_ref_t_future=self.y_ref[0]
        self.y_ref_t=0.0
        self.y_ref_t_1 = 0.0
        self.y_out_t=0.0
        self.y_out_t_1 = 0.0
        state=[self.u_t_1,self.u_t,self.y_ref_t,self.y_ref_t_future,self.y_out_t_1,self.y_out_t,
                     self.u_l[0],
                     self.y_ref_l[0], self.y_ref_l[1],0,0,0]
        self.action = np.zeros(2)
        return state



    def step(self,action):
        #pdb.set_trace()
        #self.action[1]=action
        # input time
        self.T_in[0]=self.T_in[1]
        self.T_in[1]+=self.T
        "Jianan Liu"
        self.u_t_2 = self.u_t_1
        self.u_t_1 = self.u_t
        self.u_t=float(5*action)
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, U=self.u_t, X0=self.x,params={"batch_num":self.batch_num} , return_x=True)
        #yout_lsim, T_lsim, xout_lsim = lsim(self.sys, U=self.action, T=self.T_in,X0=self.x)
        invalid=y_step[1]
        self.x = x_step[0][1]
        # pdb.set_trace()
        self.x = [float(self.x)]

        "y_out"
        self.y_out_t_1=self.y_out_t
        self.y_out_t=y_step[1]

        """
        Error serial
        """
        self.e_t_2 = self.e_t_1
        self.e_t_1 = self.e_t

        y_error=self.y_ref_t_future-y_step[1]
        self.e_t=float(y_error)

        self.y_ref_t_1 = self.y_ref_t
        self.y_ref_t = self.y_ref_t_future
        if self.N_counter == self.N:
            self.y_ref_t_future=self.y_ref[self.N-1]
            state = [self.u_t_1, self.u_t,self.y_ref_t,
                     self.y_ref_t_future,self.y_out_t_1,self.y_out_t,self.u_l[self.N_counter-1],self.y_ref_l[self.N_counter-2], self.y_ref_l[self.N_counter-1],self.u_l[self.N_counter-1],self.y_out_l[self.N_counter-2],self.y_out_l[self.N_counter-1]]
            """
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter-1], self.u_l[self.N_counter-1]]
            """
        elif self.N_counter == (self.N-1):
            self.y_ref_t_future = self.y_ref[self.N_counter]
            state = [self.u_t_1, self.u_t,self.y_ref_t,
                     self.y_ref_t_future,self.y_out_t_1,self.y_out_t,self.u_l[self.N_counter],self.y_ref_l[self.N_counter], self.y_ref_l[self.N_counter],self.u_l[self.N_counter-1],self.y_out_l[self.N_counter-2],self.y_out_l[self.N_counter-1]]

        elif self.N_counter == 1:
            self.y_ref_t_future=self.y_ref[self.N_counter]
            state = [self.u_t_1, self.u_t,self.y_ref_t,
                     self.y_ref_t_future,self.y_out_t_1,self.y_out_t,self.u_l[self.N_counter],self.y_ref_l[self.N_counter], self.y_ref_l[self.N_counter + 1],self.u_l[self.N_counter-1],0,self.y_out_l[self.N_counter-1]]


        else:
            self.y_ref_t_future=self.y_ref[self.N_counter]
            state = [self.u_t_1, self.u_t,self.y_ref_t,
                     self.y_ref_t_future,self.y_out_t_1,self.y_out_t,self.u_l[self.N_counter],self.y_ref_l[self.N_counter], self.y_ref_l[self.N_counter + 1],self.u_l[self.N_counter-1],self.y_out_l[self.N_counter-2],self.y_out_l[self.N_counter-1]]

            """
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter-2],self.e_l[self.N_counter-1],self.e_l[self.N_counter], self.u_l[self.N_counter-2],self.u_l[self.N_counter-1],self.u_l[self.N_counter],
                     self.y_ref_t[self.N_counter-1],self.y_ref_t[self.N_counter],self.y_ref_t[self.N_counter+1]]
            """

        #pdb.set_trace()
        #state=[self.e_t_2,self.e_t_1,self.e_t,self.u_t_2,self.u_t_1,self.u_t,self.y_ref_t_1,self.y_ref_t,self.y_ref_t_future,self.e_l[self.N_counter],self.u_l[self.N_counter]]
        #state = [yout_lsim[1]]
        #state=[y_error,action,self.y_re]
        #state=yout_lsim[1]
        ################
        # last batch information
        ###############

        #############################################
        if (y_error**2)<self.er_lim:
        #if 0:
            #reward=self.constant+self.ak1*(y_error**2)+self.ac_co_1*((self.u_t-self.u_t_1)**2)
            #reward = self.constant + self.ak1 * (y_error ** 2) + self.ac_co_1 * ((self.u_t - self.u_t_1) ** 2)+self.ac_co_l_1 * ((self.u_t - self.u_l[self.N_counter-1]) ** 2)
            reward = self.constant+self.ak1 * (y_error ** 2) + self.au_1 * ((self.u_t - self.u_t_1) ** 2) + self.au_l_1 * (
                    (self.u_t - self.u_l[self.N_counter - 1]) ** 2) + self.er_l_1 * (
                             (self.e_t - self.e_l[self.N_counter - 1]) ** 2)
            reward = self.scale*reward
        else:
            #reward = self.ak*(y_error**2)+self.ac_co_2* ((self.u_t-self.u_t_1)**2)
            #reward = self.ak * (y_error ** 2) + self.ac_co_2 * ((self.u_t - self.u_t_1) ** 2)+self.ac_co_l_2 * ((self.u_t - self.u_l[self.N_counter-1]) ** 2)+self.er_co_l_2 * ((self.e_t - self.e_l[self.N_counter-1]) ** 2)
            reward = self.ak * (y_error ** 2) + self.au_2 * ((self.u_t - self.u_t_1) ** 2) + self.au_l_2 * (
                        (self.u_t - self.u_l[self.N_counter - 1]) ** 2) + self.er_l_2 * (
                                 (self.e_t - self.e_l[self.N_counter - 1]) ** 2)
            reward = self.scale*reward
        self.e_l[self.N_counter-1]=self.e_t
        self.u_l[self.N_counter-1]=self.u_t
        self.y_ref_l[self.N_counter-1]=self.y_ref[self.N_counter-1]
        self.y_out_l[self.N_counter - 1] = self.y_out_t
        # done is true only after N step
        self.N_counter += 1
        if self.N_counter==(self.N+1):
            done=1
            self.batch_num += 1
        else:
            done=0
        return state,reward,done,invalid
    def close(self):
        pass








if __name__=="__main__":
    def updatestep(t, x, u, params):
        # Parameter setup

        # Map the states into local variable names
        # a=0.5+0.3*np.sin(0.1*t)
        # pdb.set_trace()
        # Compute the discrete updates
        # pdb.set_trace()
        dY = -np.sin(x) + (0.5 + 0.3 * np.sin(0.1 * t)) * u

        return [dY]

    sys = control.NonlinearIOSystem(
        updatestep, None, inputs=('u'), outputs=('x'),
        states=('x'), name='nonlinear')
    env=NonlinearSysEnv(180,sys)
    #pdb.set_trace()
    ##########################################################
    #test
    # pdb.set_trace()
    """
    def updatestep_test(t, x, u, params):
        # Parameter setup

        # Map the states into local variable names
        # a=0.5+0.3*np.sin(0.1*t)
        # pdb.set_trace()
        # Compute the discrete updates
        # pdb.set_trace()
        dY = -np.sin(x) + (0.5 + 0.3 * np.sin(0.1 * t)) * u

        return [dY]


    """
    def updatestep_test(t, x, u, params):
        # Parameter setup

        # Map the states into local variable names
        # a=0.5+0.3*np.sin(0.1*t)
        # pdb.set_trace()
        # Compute the discrete updates
        # pdb.set_trace()
        dY = -np.sin(x) + (0.5 + 0.3 ) * u

        return [dY]

    sys_test = control.NonlinearIOSystem(
        updatestep, None, inputs=('u'), outputs=('x'),
        states=('x'), name='nonlinear_test')

    X0 = [0]
    T = np.arange(0, 201, dtype=int)  # Simulation 70 years of time
    # pdb.set_trace()

    T_step = np.zeros(2)
    T_step[1] = 1
    U_step = np.zeros(2)
    U_step[0] = 0
    U_step[1] = 1
    Y_step = np.zeros(201)
    # pdb.set_trace()
    # Simulate the system
    sum = 0
    for item in range(200):
        T_step[0] = item
        T_step[1] = item + 1
        t_step, y_step, x_step = control.input_output_response(sys_test, T_step, U=1, X0=X0, return_x=True)
        Y_step[item + 1] = y_step[1]
        X0 = x_step[0][1]
        X0 = [float(X0)]
        U_step[0] = 1

    #######################################################
    Y_env = np.zeros(201)

    # test for the env
    for item in range(179):
        out=env.step(1)
        #pdb.set_trace()
        Y_env[item + 1] = out[3]




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