import os
import matplotlib.pyplot as plt   # MATLAB plotting functions
from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import control
import copy
import typing
import pprint
import csv
#np.set_printoptions(precision=20,suppress=True)
class BatchSysEnv:
    def __init__(self,T_length,sys,T=0.01,X0=np.array((0.47,396.9))):
        self.sys=sys
        self.T=T
        self.T_in=np.array((0.0,0.0))
        self.X0=copy.deepcopy(X0)
        self.X0_unchanged=copy.deepcopy(X0)
        self.T_length=T_length
        self.env_num=1
        self.batch_num = 0
        self.time=0
        # define the reference trajectory
        self.y_ref = np.ones((T_length, 2))
        self.y_ref[:, 0] = 0.57 * self.y_ref[:, 0]
        self.y_ref[:, 1] = 395 * self.y_ref[:, 1]
        #pdb.set_trace()
        #define the dim of the state space
        self.m = 2
        self.n = 2
        #self.r = 2  # No useful
        self.l = 2
        # define the 2DILC Controller
        #self.K = np.array([[-153.85851789,-63.47479029,39.07891756,39.07647162]
        #       ,[-79.97401402,-153.08904682,5.98394417,6.03835231]])
        self.K = np.array([[-136.25784389, -47.06366214, 26.72598588, 26.5998102],
                      [24.30408843, -94.52509352, -0.91989448, -0.7579483]])
        # define the equilibirum point
        self.x_qp = np.array([[0.57336624], [395.3267527]])
        self.u_qp = np.array([[1.], [0.]])
        # Increasing the dim from 1 to 2
        self.x_k = copy.deepcopy(np.expand_dims(X0,axis=0))
        'only the state need a additional time length to cal the initial ilc control'
        self.x_k_last = np.repeat(self.x_k,T_length+1,axis=0)
        self.x_k_current = np.repeat(self.x_k,T_length+1,axis=0)
        self.y_k_last = np.repeat(self.x_k,T_length+1,axis=0)
        #pdb.set_trace()
        # merge the sigma_k and e_k
        self.x_2d = np.zeros((1, self.l + self.m))
        #self.r_k = np.zeros((1, self.n))
        #self.u_k = np.zeros((1, self.n))
        self.r_k = np.zeros((self.n,1))
        self.u_k = np.zeros((self.n,1))
        self.u_k_last = np.zeros((self.T_length, self.n))
        self.delta_u_k = np.zeros((self.n, 1))
        self.delta_u_k_last = np.zeros((self.T_length,self.n))

        self.input_signal = np.zeros((self.n,1))
        #pdb.set_trace()
        # give the first 2D ILC control signal
        #self.cal_2DILCcontroller()
        #TODO: set the multi weight
        # reward function weight
        self.A1=-10.
        #self.A2=self.A1/200
        self.A2=-100
        #pdb.set_trace()
        # define the history information about the RL control signal
        self.u_rl_k_last = np.zeros((self.T_length, self.n))
        # give the first 2D ILC control signal
        self.cal_2DILCcontroller()
        # environment information
        self.env_name="MIMI_CSTR_time_batch"
        self.state_dim=28
        self.action_dim=2
        #self.max_step=N
        self.if_discrete=False
        self.target_return = 3.5
        self.episode_return = 0.0
        #self.action=np.zeros(2)
        #self.state=np.zeros((1, self.state_dim))
        self.state = np.zeros((2, 14))






    def reset(self):
        # set the sample time
        self.T_in = np.array((0.0, 0.0))
        self.X0=copy.deepcopy(self.X0_unchanged)
        self.time = 0
        self.x_k = copy.deepcopy(np.expand_dims(self.X0,axis=0))
        self.input_signal = np.zeros((self.n,1))
        self.x_k_last = copy.deepcopy(self.x_k_current)
        #self.batch_num+=1 # +1  ???
        # give the first 2D ILC control signal
        #pdb.set_trace()
        self.cal_2DILCcontroller()
        # the state space
        #pdb.set_trace()

        # 0 to 2 for the uRL
        self.state[:,0]=np.array((0.,0.))
        self.state[:,1] =np.array((0.,0.))
        self.state[:,2] = self.u_rl_k_last[self.time,:]
        # 3 to 5 for the uILC
        self.state[:,3]=np.array((0.,0.))
        self.state[:,4] =np.array((0.,0.))
        self.state[:,5] = self.u_k_last[self.time,:]
        #pdb.set_trace()
        #TODO: set the y_out set the initial state
        # 6 to 8 for the y_out
        self.state[:,6]=self.x_k[0]
        self.state[:,7] = self.x_k[0]
        self.state[:,8] = self.y_k_last[self.time,:]
        # 9 to 11 for the y_reference here is fixed
        self.state[:,9]=self.y_ref[0]
        self.state[:,10] = self.y_ref[0]
        self.state[:,11] = self.y_ref[self.time,:]
        #pdb.set_trace()
        # 12 for the next sample time y_reference
        self.state[:,12] = self.y_ref[self.time,:]
        # 13 for the current uILC
        self.state[:,13] = self.u_k[:,0]
        #pdb.set_trace()
        """squeeze the dimensions """
        state =np.reshape(self.state,self.state_dim)
        """conver the np.array to the list"""
        state=state.tolist()
        return state





    def step(self,action):
        # set the continuous sample time
        #pdb.set_trace()
        #TODO: set the action
        # map the [-1,1] to the constrained input
        #pdb.set_trace()
        action_map_u1=5*action[0]+5
        action_map_u2 = 40*action[1] -40
        #action_map_u1=action[0]
        #action_map_u2 =action[1]
        #self.T_in[0] = self.T_in[1]
        #self.T_in[1] = self.T_in[1] + self.T
        self.T_in[0] = self.time*self.T
        self.T_in[1] = (self.time+1)*self.T
        #pdb.set_trace()
        # the sum control signal of the RL+2DILC
        self.input_signal[0,0]=self.u_k[0][0]+action_map_u1
        self.input_signal[1,0] =self.u_k[1][0]+action_map_u2
        #pdb.set_trace()
        # TODO: Constained
        if self.input_signal[0,0] > 10:
            self.input_signal[0,0] = 10
        elif self.input_signal[0,0] < 0:
            self.input_signal[0,0] = 0

        # input 2 is the cooling so only negative
        if self.input_signal[1,0] < -150:
            self.input_signal[1,0] = -150
        elif self.input_signal[1,0] > 0:
            self.input_signal[1,0] = 0
        response_input = np.repeat(self.input_signal, 2, axis=1)
        #pdb.set_trace()
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, response_input, X0=self.X0,params={"batch_num":self.batch_num}, return_x=True,method='LSODA')
        """
        if self.batch_num==0 and self.time==6:
            pdb.set_trace()
        """
        # the state space
        #pdb.set_trace()
        # TODO: check the state space
        # 0 to 2 for the uRL
        self.state[0,0]=self.input_signal[0,0]-self.u_k[0][0]
        self.state[1, 0] = self.input_signal[1,0]-self.u_k[1][0]
        self.state[:,1] = self.u_rl_k_last[self.time,:]
        if self.time<(self.T_length-1):
            self.state[:,2] = self.u_rl_k_last[(self.time+1),:]
        else:
            self.state[:,2] = self.u_rl_k_last[self.time,:]
        # 3 to 5 for the uILC
        self.state[:,3]=self.u_k[:,0]
        self.state[:,4] = self.u_k_last[self.time,:]
        if self.time < (self.T_length - 1):
            self.state[:,5] = self.u_k_last[(self.time+1),:]
        else:
            self.state[:,5] = self.u_k_last[self.time,:]
        # 6 to 8 for the y_out
        #pdb.set_trace()
        self.state[:,6]=y_step[:,1]
        self.state[:,7] = self.y_k_last[self.time,:]
        if self.time < (self.T_length - 1):
            self.state[:,8] = self.y_k_last[(self.time+1),:]
        else:
            self.state[:,8] = self.y_k_last[self.time,:]
        # 9 to 11 for the y_reference here is fixed
        self.state[:,9]=self.y_ref[self.time,:]
        self.state[:,10] = self.y_ref[self.time,:]
        if self.time < (self.T_length - 1):
            self.state[:,11] = self.y_ref[(self.time+1),:]
        else:
            self.state[:,11] = self.y_ref[self.time,:]
        # 12 for the next sample time y_reference
        if self.time < (self.T_length - 1):
            self.state[:,12] = self.y_ref[(self.time+1),:]
        else:
            self.state[:,12] = self.y_ref[self.time,:]
        #pdb.set_trace()
        #pdb.set_trace()
        # change the initial state
        self.X0[0] = x_step[0][1]
        self.X0[1] = x_step[1][1]
        #pdb.set_trace()
        # save the data into the memory
        # ILC data
        self.u_k_last[self.time,:]=self.u_k[:,0]
        #self.delta_u_k_last[item,0] = self.u_k[0,0]-self.u_qp[0,0]
        #self.delta_u_k_last[item,1] = self.u_k[1,0]-self.u_qp[1,0]
        self.delta_u_k_last[self.time,0] = self.input_signal[0,0]-self.u_qp[0,0]
        self.delta_u_k_last[self.time,1] = self.input_signal[1,0]-self.u_qp[1,0]
        self.y_k_last[self.time,:]=y_step[:,1]
        #pdb.set_trace()
        # change the current state
        #pdb.set_trace()
        self.x_k_current[(self.time + 1),:] = x_step[:,1]
        self.x_k[0] = x_step[:,1]  # change the current information
        # RL data
        #pdb.set_trace()
        self.u_rl_k_last[self.time,0]=self.input_signal[0,0]-self.u_k[0][0]
        self.u_rl_k_last[self.time, 1] = self.input_signal[1,0]-self.u_k[1][0]
        # cal the reward fucntion
        reward= self.A1*abs(self.y_ref[self.time,0]-y_step[0,1])+self.A2*abs(self.y_ref[self.time,1]-y_step[1,1])
        #pdb.set_trace()
        reward=np.float64(reward)

        # the current time
        self.time+=1
        if self.time==200:
            self.batch_num += 1  # +1  ???
            done=1
        else:
            done=0
        if self.time<200:
            #cal the 2DILC control signal
            self.cal_2DILCcontroller()
        # 13 for the current uILC
        self.state[:,13] = self.u_k[:,0]
        invalid=1
        #pdb.set_trace()
        """squeeze the dimensions """
        state =np.reshape(self.state,self.state_dim)
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
        #pdb.set_trace()
        #cal the control signal of the 2D ILC
        #self.r_k=self.K@self.x_2d.T
        self.r_k = np.dot(self.K,self.x_2d.T)
        #pdb.set_trace()
        #self.u_k[0] = self.u_k_last[self.time] + self.r_k[0]
        #self.u_k[0]= self.u_k_last[self.time]+ self.r_k[0]+self.u_rl_k_last[self.time]
        #self.delta_u_k[0,0]=self.delta_u_k_last[self.time,0]+self.r_k[0,0]
        #self.delta_u_k[1, 0] = self.delta_u_k_last[self.time, 1] + self.r_k[1, 0]

        #self.delta_u_k[0,0]=self.delta_u_k_last[self.time,0]+self.r_k[0,0]+ self.u_rl_k_last[self.time,0]
        #self.delta_u_k[1, 0] = self.delta_u_k_last[self.time, 1] + self.r_k[1, 0]+ self.u_rl_k_last[self.time,1]
        self.delta_u_k[0,0]=self.delta_u_k_last[self.time,0]+self.r_k[0,0]
        self.delta_u_k[1, 0] = self.delta_u_k_last[self.time, 1] + self.r_k[1, 0]
        #self.u_k[0,0] = self.u_k_last[self.time,0] + self.r_k[0,0] + self.u_rl_k_last[self.time,0]
        #self.u_k[1,0] = self.u_k_last[self.time,1] + self.r_k[1,0] + self.u_rl_k_last[self.time,1]

        self.u_k[0, 0] =  self.delta_u_k[0, 0] + self.u_qp[0,0]
        self.u_k[1, 0] =  self.delta_u_k[1, 0]+ self.u_qp[1,0]
        #TODO: set the constrian
        # constained the input
        #pdb.set_trace()
        """
        if self.batch_num == 3 and self.time == 92:
            pdb.set_trace()
        """
        if self.u_k[0, 0] > 10:
            self.u_k[0, 0] = 10
        elif self.u_k[0, 0] < 0:
            self.u_k[0, 0] = 0

        # input 2 is the cooling so only negative
        if self.u_k[1, 0] < -150:
            self.u_k[1, 0] = -150
        elif self.u_k[1, 0] > 0:
            self.u_k[1, 0] = 0
        #self.delta_u_k[0,0]=self.u_k[0, 0]-self.u_qp[0,0]
        #self.delta_u_k[1, 0] =self.u_k[1, 0]-self.u_qp[1,0]







if __name__=="__main__":
    "1. define the CSTR nonlinear model"
    "From the reference Chi R, Huang B, Hou Z, et al. Data‐driven high‐order terminal iterative learning control with a " \
    "faster convergence speed[J]. International Journal of Robust and Nonlinear Control, 2018, 28(1): 103-119."
    # define the dimensions of the state space
    m = 2  # dimension of the state
    n = 2  # input
    # r=2# No useful
    l = 2  # output
    T_length = 200
    # T_length=300
    batch = 20
    save_figure = False


    def state_update(t, x, u, params):
        batch_num = params.get('batch_num', 0)
        # Map the states into local variable names
        batch_num = params.get('batch_num', 0)
        # print(batch_num)
        z1 = np.array([x[0]])
        z2 = np.array([x[1]])
        n1 = np.array([u[0]])
        n2 = np.array([u[1]])
        # Compute the discrete updates
        a = 1 + 0.1 * np.sin(2.5 * t * np.pi) + 0.1 * np.sin(batch_num * np.pi / 10)
        # a = 1+ 0.1 * np.sin(batch_num * np.pi / 10)
        dz1 = -(a + 7.2 * np.power(10., 10) * np.exp(-np.power(10., 4) / z2)) * z1 + n1
        # dz2 = -1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - z2 + 1476.946
        dz2 = 1.44 * np.power(10., 13) * np.exp(-np.power(10., 4) / z2) * z1 - a * z2 + 0.041841 * n2 + 310 * a
        # pdb.set_trace()
        return [dz1, dz2]


    def ouput_update(t, x, u, params):
        # Parameter setup

        # Compute the discrete updates
        y1 = x[0]
        y2 = x[1]

        return [y1, y2]


    Nonlinear_CSTR = control.NonlinearIOSystem(
        state_update, ouput_update, inputs=('u1', 'u2'), outputs=('y1', 'y2'),
        states=('dz1', 'dz2'), dt=0, name='Nonlinear_CSTR')

    print("Continuous system:", control.isctime(Nonlinear_CSTR))
    "2. define the initial state "
    X0 = np.array((0.47, 396.9))  # Initial x1, x2
    T = np.array((0.0, 0.01))
    sample_time = 0.01
    x_k = copy.deepcopy(np.expand_dims(X0, axis=0))
    controlled_system=BatchSysEnv(T_length=T_length,sys=Nonlinear_CSTR,X0=X0)
    #controlled_system.reset()
    out=np.repeat(x_k,T_length+1,axis=0)
    out_batch=[]
    for batch_index in range(batch):
        for item in range(T_length):
            state,out_tem,done,invalid=controlled_system.step([0.0,0.0])
            #pdb.set_trace()
            out[(item+1),0]=state[6]
            out[(item + 1), 1] = state[20]
        out_batch.append(out)
        #pdb.set_trace()
        controlled_system.reset()
        out = np.repeat(x_k,T_length+1,axis=0)
    pdb.set_trace()
    """save the sqrt to the csv"""
    with open('compare/y_out_rl_env_float20_new.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows

        writer.writerows(map(lambda x: [x], out_batch))
    pdb.set_trace()
    a=2