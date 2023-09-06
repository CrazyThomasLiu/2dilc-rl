import sys
import os
import pdb
import control
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
class  BatchSysEnv:
    def __init__(self):
        'set the hyperparameters'
        #define the dim of the state space
        self.m = 2
        self.n = 1
        self.r = 1  # No useful None
        self.l = 1
        # sample frequency
        self.T=1.0
        # initial sample time
        self.T_in=np.array((0.0,0.0))
        # initial state
        self.X0=np.array((0.0, 0.0))
        # duration steps per batch
        self.T_length=200
        # batch counter
        self.batch_num = 0
        # time counter
        self.time=0
        'define the batch system'
        def state_update(t, x, u, params):
            # get the parameter from the params
            batch_num = params.get('batch_num', 0)
            # Parameter setup
            # non-repetitive nature
            #sigma1 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            #sigma2 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            #sigma3 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            #sigma4 = 0.5 * np.sin(batch_num * 2 * np.pi / 10)
            # model mismatch
            #sigma1 += 0.5 * np.sin(0.1 * t)
            #sigma2 += 0.5 * np.sin(0.1 * t)
            #sigma3 += 0.5 * np.sin(0.1 * t)
            #sigma4 += 0.5 * np.sin(0.1 * t)
            # Map the states into local variable names
            z1 = np.array([x[0]])
            z2 = np.array([x[1]])
            #z3 = np.array([x[2]])
            # Compute the discrete updates
            dz1 = 1.607 * z1 + z2 +1.239* u
            dz2 = -0.6086 * z1 -0.9282* u
            # pdb.set_trace()
            return [dz1, dz2]

        def ouput_update(t, x, u, params):
            # Parameter setup

            # Compute the discrete updates
            y = x[0]

            return [y]

        self.sys = control.NonlinearIOSystem(
            state_update, ouput_update, inputs=('u'), outputs=('y'),
            states=('dz1', 'dz2'), dt=1, name='injection_modeling')

        self.input_signal = np.zeros((self.n,1))

    def reset(self):
        # reset the initial sample time
        self.T_in = np.array((0.0, 0.0))
        # reset the initial state
        self.X0=np.array((0.0, 0.0))
        # reset the time
        self.time = 0
        output = self.X0[0]
        return output


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
        output=self.X0[0]
        #pdb.set_trace()
        return output



if __name__=="__main__":
    # set the simulation batches
    batch = 50
    T_length = 200
    # PI control parameters
    #K_p=1.05394935
    #K_i=0.060275608914548626
    K_p=0.6104
    K_i=0.0037
    # robust control parameters
    #L1=-0.05409614
    #L2=1.08910147e-10
    #L3=0.24500174
    L1=-0.0060
    L2=-2.8726e-21
    L3=0.6944
    # define the reference trajectory
    y_ref = 200 * np.ones((T_length, 1))
    y_ref[100:] = 1.25 * y_ref[100:]
    e_k_list=np.zeros(T_length+1)
    e_k_1_list = np.zeros(T_length+1)
    # define the sum of the e_{s} not the e
    # the initial e_s_sum=0
    'for e_s_sum 0=-1  1=0'
    e_s_k_sum_list=np.zeros(T_length+1)
    e_s_k_1_sum_list = np.zeros(T_length+1)
    # define the y_s
    # the initial y_s=0
    ys_k_list=np.zeros(T_length)
    ys_k_1_list = np.zeros(T_length)
    # initial the controlled system
    controlled_system = BatchSysEnv()
    #
    RMSE=np.zeros(batch)
    for batch_index in range(batch):
        # reset the sum of the current error
        e_s_sum=0
        y_current = controlled_system.reset()
        # e
        e_current = 0.0-y_current
        e_k_list[0] = copy.deepcopy(e_current)
        #pdb.set_trace()
        for item in range(T_length):
            # e_sum_current
            #delta_e_s
            """
            if item==0:
                delta_e_s=0.0
            else:
                delta_e_s=e_s_k_sum_list[item-1]-e_s_k_1_sum_list[item-1]
            """
            delta_e_s = e_s_k_sum_list[item] - e_s_k_1_sum_list[item]
            'y_s'
            y_s=ys_k_1_list[item]+L1*delta_e_s+L2*e_current+L3*e_k_1_list[item+1]
            ys_k_list[item]=copy.deepcopy(y_s)
            #e_s
            e_s=y_s-y_current
            #e_s_sum
            e_s_sum=e_s_sum+e_s
            'using the last time'
            e_s_k_sum_list[item+1]=copy.deepcopy(e_s_sum)
            #pdb.set_trace()
            #u
            u=K_p*e_s+K_i*e_s_sum
            y_current = controlled_system.step(u)
            # e
            e_current = y_ref[item] - y_current
            e_k_list[item+1] = copy.deepcopy(e_current)
            #pdb.set_trace()
        e_k_1_list=copy.deepcopy(e_k_list)
        e_s_k_1_sum_list=copy.deepcopy(e_s_k_sum_list)
        ys_k_1_list=copy.deepcopy(ys_k_list)
        """
        #calculation of the RMSE
        tem=0.0
        for time in range(T_length):
            tem += (abs(e_k_list[time+1])) ** 2
            RMSE[batch_index]=math.sqrt(tem / T_length)
        """
        #calculation of the ATE
        tem=0.0
        for time in range(T_length):
            tem += abs(e_k_list[time+1])
            RMSE[batch_index]=tem / T_length
        #print(e_k_list)
        #pdb.set_trace()
    #print(e_k_list)

    pdb.set_trace()
    "2. Plot of the sum RMSE"
    batch_time = range(1, batch + 1)
    fig = plt.figure(figsize=(9, 6.5))
    x_major_locator = MultipleLocator(int(batch / 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    #plt.plot(batch_time, y_2dilc_show, linewidth=2, color='tab:blue', linestyle='dashdot')
    plt.plot(batch_time, RMSE, linewidth=2, color='tab:orange', linestyle='solid')
    plt.grid()

    xlable = 'Batch:$\mathit{k} $'
    ylable = 'RMSE:$\mathit{I_{k}}$'
    font2_rmse = {'family': 'Arial',
                  'weight': 'bold',
                  'size': 22,
                  }
    font2_legend = {'family': 'Arial',
                    'weight': 'bold',
                    'size': 16,
                    }
    plt.xlabel(xlable, font2_rmse)
    plt.ylabel(ylable, font2_rmse)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(['2D Iterative Learning Controller', '2D ILC-RL Control Scheme'], prop=font2_legend)
    #if save_figure == True:
    #    plt.savefig('Linear_injection_molding_compare_rmse.pdf')

    plt.show()
    a=2