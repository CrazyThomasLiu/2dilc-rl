import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np

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

class LinearSysEnv:
    def __init__(self,N,sys,ak=-10,T=1,X0=[0.,0.],):
        self.sys=sys
        self.T=T
        self.T_in=np.zeros(2)
        self.T_in[1]=self.T
        self.X0=np.array(X0)
        self.N=N
        self.env_num=1
        """
        weight
        sinus

        """
        self.er_lim=0.0001
        #第一个公式
        self.constant=1
        #self.ac_co_1=-10
        #self.ak1=-50
        self.ak1 = 0
        self.ac_co_1 = -0.1
        self.ac_co_l_1 = -0.1


        #第二个公式
        self.ak=-10  ###weight
        self.ac_co_2=-0.1
        self.ac_co_l_2 = -0.1
        self.scale=0.8
        ###undefined varaible


        ###undefined varaible
        """
        self.y_ref = np.zeros(N)
        for item in range(N):
            self.y_ref[item] = (np.sin((item) * 2 * np.pi / N)) ** 10000

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
        self.env_name="Linearsys_SISO_v2"
        #self.state_dim=10000
        self.state_dim=16
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
        self.zero=0.0






    def reset(self):
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
        state=[self.e_t_2,self.e_t_1,self.e_t,self.u_t_2,self.u_t_1,self.u_t,self.y_ref_t_1,self.y_ref_t,self.y_ref_t_future,self.zero,
                     self.e_l[0], self.zero,
                     self.u_l[0],
                     self.zero, self.y_ref_l[0], self.y_ref_l[1]]
        self.action = np.zeros(2)
        return state



    def step(self,action):
        #pdb.set_trace()
        #self.action[1]=action
        "Jianan Liu"
        self.u_t_2 = self.u_t_1
        self.u_t_1 = self.u_t
        self.u_t=float(2*action)
        self.action[0] = self.u_t
        self.action[1] = self.u_t
        #pdb.set_trace()
        yout_lsim, T_lsim, xout_lsim = lsim(self.sys, U=self.action, T=self.T_in,X0=self.x)
        self.x=xout_lsim[1,:]
        invalid=xout_lsim
        """
        Error serial
        """
        self.e_t_2 = self.e_t_1
        self.e_t_1 = self.e_t

        y_error=self.y_ref_t_future-yout_lsim[1]
        self.e_t=float(y_error)

        self.y_ref_t_1 = self.y_ref_t
        self.y_ref_t = self.y_ref_t_future
        if self.N_counter == self.N:
            self.y_ref_t_future=self.y_ref[self.N-1]
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter - 2],
                     self.e_l[self.N_counter-1], self.u_l[self.N_counter - 2],
                     self.u_l[self.N_counter-1],
                     self.y_ref_l[self.N_counter - 2], self.y_ref_l[self.N_counter-1], self.y_ref_l[self.N_counter -1]]
            """
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter-1], self.u_l[self.N_counter-1]]
            """
        elif self.N_counter == (self.N-1):
            self.y_ref_t_future = self.y_ref[self.N_counter]
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter - 1],
                     self.e_l[self.N_counter], self.u_l[self.N_counter - 1],
                     self.u_l[self.N_counter],
                     self.y_ref_l[self.N_counter - 1], self.y_ref_l[self.N_counter], self.y_ref_l[self.N_counter]]

        else:
            self.y_ref_t_future=self.y_ref[self.N_counter]
            state = [self.e_t_2, self.e_t_1, self.e_t, self.u_t_2, self.u_t_1, self.u_t, self.y_ref_t_1, self.y_ref_t,
                     self.y_ref_t_future, self.e_l[self.N_counter - 1],
                     self.e_l[self.N_counter], self.u_l[self.N_counter - 1],
                     self.u_l[self.N_counter],
                     self.y_ref_l[self.N_counter - 1], self.y_ref_l[self.N_counter], self.y_ref_l[self.N_counter + 1]]

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
            #reward=self.constant+self.ak1*(y_error**2)+self.ac_co_1*((self.u_t-self.u_t_1)**2)
            reward = self.constant + self.ak1 * (y_error ** 2) + self.ac_co_1 * ((self.u_t - self.u_t_1) ** 2)+self.ac_co_l_1 * ((self.u_t - self.u_l[self.N_counter-1]) ** 2)
            reward = self.scale*reward
        else:
            #reward = self.ak*(y_error**2)+self.ac_co_2* ((self.u_t-self.u_t_1)**2)
            reward = self.ak * (y_error ** 2) + self.ac_co_2 * ((self.u_t - self.u_t_1) ** 2)+self.ac_co_l_2 * ((self.u_t - self.u_l[self.N_counter-1]) ** 2)
            reward = self.scale*reward
        self.e_l[self.N_counter-1]=self.e_t
        self.u_l[self.N_counter-1]=self.u_t
        self.y_ref_l[self.N_counter-1]=self.y_ref[self.N_counter-1]
        # done is true only after N step
        self.N_counter += 1
        if self.N_counter==(self.N+1):
            done=1
        else:
            done=0
        return state,reward,done,invalid
    def close(self):
        pass








if __name__=="__main__":

    num=[0.4]
    den=[1,0.24,0.16]
    sys2=tf(num,den)
    sys3=tf2ss(sys2)
    #pdb.set_trace()
    env=LinearSysEnv(180,sys3)
    ########################################################
    ###test for this class
    out=env.step(1)
    pdb.set_trace()
    a=2