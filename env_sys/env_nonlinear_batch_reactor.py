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
    def __init__(self,T_length,sys,T=0.01,X0=np.array((0.5, 310.)),action_co=10000.):
        self.sys=sys
        self.T=T
        self.T_in=np.array((0.0,0.0))
        self.X0 = copy.deepcopy(X0)
        self.X0_unchanged = copy.deepcopy(X0)
        self.T_length=T_length
        self.env_num=1
        self.batch_num = 0
        self.time=0
        self.action_co=action_co
        # define the reference trajectory
        self.y_ref = np.ones((self.T_length, 1))
        self.y_ref[0:100, 0] = 370 * self.y_ref[0:100, 0]
        self.y_ref[100:T_length, 0] = 350 * self.y_ref[100:T_length, 0]
        #define the dim of the state space
        self.m = 2
        self.n = 1
        self.r = 1  # No useful
        self.l = 1
        # define the 2DILC Controller
        self.K = np.array([[1185.5239983, -2045.16033935,1017.79150778]])
        # define the equilibirum point
        self.x_qp = np.array([[0.48632751], [350.]])
        self.u_qp = np.array([[890.6456]])



        # Increasing the dim from 1 to 2
        self.x_k = copy.deepcopy(np.expand_dims(X0,axis=0))
        'only the state need a additional time length to cal the initial ilc control'
        self.x_k_last = np.repeat(self.x_k,T_length+1,axis=0)
        self.x_k_current = np.repeat(self.x_k,T_length+1,axis=0)
        self.y_k_last = np.repeat(np.array([[310.]]), T_length, axis=0)
        # merge the sigma_k and e_k
        self.x_2d = np.zeros((1, self.l + self.m))
        self.r_k = np.zeros((1, self.n))
        self.u_k = np.zeros((1, self.n))
        self.u_k_last = np.repeat(np.array([[self.u_qp[0,0]]]),T_length,axis=0)
        self.delta_u_k = np.zeros((self.n, 1))
        self.delta_u_k_last = np.zeros((self.T_length,self.n))
        self.input_signal = np.zeros((self.n,1))
        # give the first 2D ILC control signal
        #self.cal_2DILCcontroller()

        # reward function weight
        self.A=-40.
        # define the history information about the RL control signal
        self.u_rl_k_last = np.zeros((self.T_length, self.n))
        # give the first 2D ILC control signal
        self.cal_2DILCcontroller()
        # environment information
        self.env_name="SISO_CSTR"
        #self.state_dim=10000
        self.state_dim=14
        self.action_dim=1
        #self.max_step=N
        self.if_discrete=False
        self.target_return = 0.0
        self.episode_return = 0.0
        self.max_step = 300
        #self.action=np.zeros(2)
        self.state=np.zeros((1, self.state_dim))






    def reset(self):
        # set the sample time
        self.T_in = np.array((0.0, 0.0))
        self.X0=copy.deepcopy(self.X0_unchanged)
        self.time = 0
        self.x_k = copy.deepcopy(np.expand_dims(self.X0,axis=0))
        self.input_signal = np.zeros((self.n,1))
        self.x_k_last = copy.deepcopy(self.x_k_current)
        # give the first 2D ILC control signal
        self.cal_2DILCcontroller()
        # the state space
        # 0 to 2 for the uRL
        self.state[0][0]=0.
        self.state[0][1] =0.
        self.state[0][2] = self.u_rl_k_last[self.time][0]/100

        #TODO the initial input of the ILC is the eq u
        # 10000 to 5 for the uILC
        self.state[0][3] =self.u_qp[0,0]/100
        self.state[0][4] =self.u_qp[0,0]/100
        self.state[0][5] = self.u_k_last[self.time][0]/100
        # 6 to 8 for the y_out
        self.state[0][6]=self.x_k[0,1]
        self.state[0][7] =self.x_k[0,1]
        self.state[0][8] = self.y_k_last[self.time][0]
        # 9 to 11 for the y_reference here is fixed
        self.state[0][9]=self.y_ref[0,0]
        self.state[0][10] = self.y_ref[0,0]
        self.state[0][11] = self.y_ref[self.time][0]
        self.state[0][12] = self.y_ref[self.time][0]
        # 13 for the current uILC
        self.state[0][13] = self.u_k[0]/100
        """squeeze the dimensions """
        state =np.squeeze(self.state)
        """conver the np.array to the list"""
        state=state.tolist()
        return state





    def step(self,action):
        # set the continuous sample time
        """here how to choose the action"""
        co_action=self.action_co*action
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T
        # the sum control signal of the RL+2DILC
        self.input_signal[0,0]=self.u_k[0,0]+co_action
        response_input = np.repeat(self.input_signal, 2, axis=1)
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, response_input, X0=self.X0,params={"batch_num":self.batch_num}, return_x=True)
        # the state space
        # 0 to 2 for the uRL
        self.state[0][0] = co_action/100
        self.state[0][1] = self.u_rl_k_last[self.time][0]/100
        if self.time<(self.T_length-1):
            self.state[0][2] = self.u_rl_k_last[self.time+1][0]/100
        else:
            self.state[0][2] = self.u_rl_k_last[self.time][0]/100
        self.state[0][3]=self.u_k[0,0]/100
        self.state[0][4] = self.u_k_last[self.time][0]/100
        if self.time < (self.T_length - 1):
            self.state[0][5] = self.u_k_last[self.time+1][0]/100
        else:
            self.state[0][5] = self.u_k_last[self.time][0]/100
        # 6 to 8 for the y_out
        self.state[0][6]=y_step[1]
        self.state[0][7] = self.y_k_last[self.time][0]
        if self.time < (self.T_length - 1):
            self.state[0][8] = self.y_k_last[self.time+1][0]
        else:
            self.state[0][8] = self.y_k_last[self.time][0]
        # 9 to 11 for the y_reference here is fixed
        self.state[0][9]=self.y_ref[self.time][0]
        self.state[0][10] = self.y_ref[self.time][0]
        if self.time < (self.T_length - 1):
            self.state[0][11] = self.y_ref[self.time+1][0]
        else:
            self.state[0][11] = self.y_ref[self.time][0]
        # 10000_constrain_re5 for the next sample time y_reference
        if self.time < (self.T_length - 1):
            self.state[0][12] = self.y_ref[self.time+1][0]
        else:
            self.state[0][12] = self.y_ref[self.time][0]
        # change the initial state
        self.X0[0] = x_step[0][1]
        self.X0[1] = x_step[1][1]
        # save the data into the memory
        # ILC data
        self.u_k_last[self.time]=self.u_k[0,0]
        self.y_k_last[self.time]=y_step[1]

        self.delta_u_k_last[self.time,0] = self.input_signal[0,0]-self.u_qp[0,0]

        for item1 in range(self.m):
            self.x_k_current[(self.time+1)][item1]=x_step[item1][1]
            self.x_k[0][item1]=x_step[item1][1]   #change the current information
        # RL data
        self.u_rl_k_last[self.time,0]=co_action
        # cal the reward fucntion
        reward= self.A*abs(self.y_ref[self.time]-y_step[1])
        reward=np.float64(reward)

        # the current time
        self.time+=1
        if self.time==300:
            self.batch_num += 1
            done=1
        else:
            done=0
        if self.time<300:
            self.cal_2DILCcontroller()
        # 13 for the current uILC
        self.state[0][13] = self.u_k[0,0]/100
        invalid=self.input_signal[0,0]
        """squeeze the dimensions """
        state =np.squeeze(self.state)
        """conver the np.array to the list"""
        state=state.tolist()
        return state,reward,done,invalid

    def close(self):
        pass

    def cal_2DILCcontroller(self):
        # cal the initial 2D system state matrix
        tem_x=self.x_k[0]-self.x_k_last[self.time]  # 上一个批次应该是0
        tem_y=self.y_ref[self.time]-self.y_k_last[self.time]
        self.x_2d=np.block([[tem_x,tem_y]])
        #cal the control signal of the 2D ILC
        self.r_k = np.dot(self.K,self.x_2d.T)
        self.delta_u_k[0,0]=self.delta_u_k_last[self.time,0]+self.r_k[0,0]
        self.u_k[0, 0] =  self.delta_u_k[0, 0] + self.u_qp[0,0]

        ### The constrain is not added, because only the sum of the ILC and RL signal is applied to the actual system.

        """
        if self.u_k[0, 0] > 100000:
            self.u_k[0, 0] = 100000
        elif self.u_k[0, 0] < -100000:
            self.u_k[0, 0] = -100000
        """
        #self.u_k[0]= self.u_k_last[self.time]+ self.r_k[0]+self.u_rl_k_last[self.time]

    def save_or_load_history_implementation(self, cwd, if_save):
        save_path = f"{cwd}/Ilc_data.npz"
        if_load = None
        if if_save:

            np.savez_compressed(save_path, x_k_last=self.x_k_last, y_k_last=self.y_k_last,u_k_last=self.u_k_last, u_rl_k_last=self.u_rl_k_last,x_k_current=self.x_k_current,\
            delta_u_k_last=self.delta_u_k_last,)
            print(f"| ILC data save in: {save_path}")
        elif os.path.isfile(save_path):
            ILC_data_dict = np.load(save_path)
            self.x_k_last = ILC_data_dict['x_k_last']
            self.y_k_last = ILC_data_dict['y_k_last']
            self.u_k_last = ILC_data_dict['u_k_last']
            self.u_rl_k_last = ILC_data_dict['u_rl_k_last']
            self.x_k_current = ILC_data_dict['x_k_current']
            self.delta_u_k_last = ILC_data_dict['delta_u_k_last']
            print(f"| ILC data load: {save_path}")
            if_load = True
        else:
            if_load = False
        return if_load
