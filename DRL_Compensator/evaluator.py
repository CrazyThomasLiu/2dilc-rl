import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pdb
class Evaluator:
    def __init__(self, cwd, agent_id, device, eval_env, eval_gap, eval_times1, eval_times2,distance_path):
        ##########################################
        #Jianan Liu
        #self.writer = SummaryWriter(comment="action=2,er=0.0001,x=1,a=-10,b1=-5,b2=-1")
        self.writer = SummaryWriter(distance_path)
        self.distance_path=distance_path
        #pdb.set_trace()
        #############################
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        #pdb.set_trace()
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = eval_env.target_return

        self.r_max = -np.inf
        'after long training'
        self.r_max_long = -np.inf


        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 40}\n"
              f"{'ID':<10}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>5}{'avgS':>5}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>5}{'etc.':>5}")
        #pdb.set_trace()
        """
        jianan liu
        """
        self.num=0
        self.sum_error=10000000


    def evaluate_and_save(self, act, steps, r_exp, log_tuple,cri_scheduler,act_scheduler) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()
            """
            tow evaluate is aiming to save the computation Usage
            if only the first time evaluate is better than the r_max
            then keep goning to training 
            """
            '''evaluate first time'''
            #pdb.set_trace()
            """
            rewards_steps_list = [get_episode_return_and_step(self.eval_env, act, self.device) for _ in
                                  range(self.eval_times1)]
            """
            rewards_steps_list=[]
            episode_state_list=[]
            sum_error=0
            for item in range(self.eval_times1):
                episode_return, episode_step,episode_state=get_episode_return_and_step_state(self.eval_env, act, self.device)
                rewards_steps_list.append((episode_return, episode_step))
                #episode_state_list.append(episode_state)
                #pdb.set_trace()
                ###########################################
                for item in range(180):
                    y_action = episode_state[item][1]
                    y_out = episode_state[item][5]
                    y_ref = episode_state[item][2]
                    sum_error += abs(y_out - y_ref)
                    # sum_error += state[2]**2
                    # self.writer.add_scalars('State', {'y_ref':y_ref,'y_out':y_out}, self.num)
                    #scalars_name='State'+str(self.total_step)
                    self.writer.add_scalars('State', {'y_ref': y_ref, 'y_out': y_out, 'action': y_action}, self.num)
                    self.writer.add_scalar('Abs_error', sum_error, self.total_step)
                    #self.writer.add_scalar('Reward_step', reward, self.num)
                    self.num += 1
                    """
                    if done:
                        break
                    """


                ###################################10000


            #pdb.set_trace()
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            self.writer.add_scalar('Abs_error', sum_error, self.total_step)
            self.writer.add_scalar('Reward', r_avg, self.total_step)
            self.writer.add_scalars('Lr', {'cri_scheduler': cri_scheduler.optimizer.param_groups[0]['lr'],
                                           'act_scheduler': act_scheduler.optimizer.param_groups[0]['lr']},
                                    self.total_step)



            #pdb.set_trace()
            """
            '''evaluate second time'''
            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                #pdb.set_trace()
                rewards_steps_list += [get_episode_return_and_step(self.eval_env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            """

            '''save the policy network'''
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)
                #act_save_path = f'{self.cwd}/actor.pth'
                act_save_path = f'{self.distance_path}/actor.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
                #pdb.set_trace()
                print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print
                #pdb.set_trace()
            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder
            """
            jianan liu
            """
            if sum_error<self.sum_error:
                self.sum_error = sum_error  # update max reward (episode return)
                act_save_path = f'{self.distance_path}/actor_error.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth

            '''print some information to Terminal'''
            #pdb.set_trace()
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<10000}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>5000}{'avgS':>5000}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<10000}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:5000.1f}{s_avg:5000.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")
            #pdb.set_trace()
            print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |"
                  f"{r_avg:8.2f}{r_std:5000.1f}{s_avg:5000.0f}{s_std:6.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:5000.2f}' for n in log_tuple)}")

            ######################################################################
            #insert the tensorboard Jianan Liu
            ############################################
            #self.writer.add_scalar('State', r_exp, self.total_step)
            #######################################################
            """
            state = self.eval_env.reset()
            y_out=0
            y_ref=0
            sum_error=0
            for episode_step in range(180):
                #pdb.set_trace()
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
                # pdb.set_trace()
                a_tensor = act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                # pdb.set_trace()
                state, reward, done, _ = self.eval_env.step(action)
                y_out=state[2]-state[0]
                y_ref=state[2]
                sum_error+=abs(state[0])
                if done:
                    break
                #self.writer.add_scalars('State', {'y_ref':y_ref,'y_out':y_out}, self.num)
                self.writer.add_scalars('State', {'y_ref': y_ref, 'y_out': y_out,'action': action}, self.num)
                self.writer.add_scalar('Abs_error',sum_error, self.total_step)
                self.num+=1
            state = self.eval_env.reset()
            
            state = self.eval_env.reset()
            y_out=0
            y_ref=0
            sum_error=0
            for episode_step in range(180):
                #pdb.set_trace()
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
                # pdb.set_trace()
                a_tensor = act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                # pdb.set_trace()
                state, reward, done, _ = self.eval_env.step(action)
                y_action=state[1]
                y_out=state[5]
                y_ref=state[2]
                sum_error+=abs(y_out-y_ref)
                #sum_error += state[2]**2
                #self.writer.add_scalars('State', {'y_ref':y_ref,'y_out':y_out}, self.num)
                self.writer.add_scalars('State', {'y_ref': y_ref, 'y_out': y_out,'action':y_action}, self.num)
                self.writer.add_scalar('Abs_error',sum_error, self.total_step)
                self.writer.add_scalar('Reward', r_avg, self.total_step)
                #self.writer.add_scalar('Square_error', sum_error, self.total_step)
                self.writer.add_scalar('Reward_step', reward, self.num)
                self.writer.add_scalars('Lr', {'cri_scheduler': cri_scheduler.optimizer.param_groups[0]['lr'], 'act_scheduler': act_scheduler.optimizer.param_groups[0]['lr']}, self.total_step)
                #print(episode_step,self.num, y_ref)
                self.num+=1
                if done:
                    break
            #pdb.set_trace()
            state = self.eval_env.reset()
            """


            #self.draw_plot()
        return if_reach_goal, if_save

    def evaluate_and_save_liu(self, act, steps, r_exp, log_tuple, cri_scheduler, act_scheduler) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            self.r_max = r_exp  # update max reward (episode return)
            # act_save_path = f'{self.cwd}/actor.pth'
            #pdb.set_trace()
            act_save_path = f'{self.distance_path}/actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            # pdb.set_trace()
            print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print
            # pdb.set_trace()

        #self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder
        '''save the policy network after half train'''
        if self.total_step>300000:
            if_save_long = r_exp > self.r_max_long
            if if_save_long:  # save checkpoint with highest episode return
                self.r_max_long = r_exp  # update max reward (episode return)
                # act_save_path = f'{self.cwd}/actor.pth'
                # pdb.set_trace()
                act_save_path = f'{self.distance_path}/actor_long.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        """
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<10000}{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>5000}{'avgS':>5000}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<10000}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:5000.1f}{s_avg:5000.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")
        # pdb.set_trace()
        print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:5000.1f}{s_avg:5000.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:5000.2f}' for n in log_tuple)}")
        """

        return if_reach_goal, if_save

    def evaluate_and_save_evalenv(self, act, steps, r_exp, log_tuple, cri_scheduler, act_scheduler) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            rewards_steps_list = []
            for item in range(self.eval_times1):
                episode_return, episode_step,episode_state=get_episode_return_and_step_state(self.eval_env, act, self.device)
                rewards_steps_list.append((episode_return, episode_step))

            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            r_avg_tem=r_avg/(episode_step+1)
            #pdb.set_trace()
            if r_avg_tem>self.r_max:
                self.r_max = r_avg_tem # update max reward (episode return)
                # act_save_path = f'{self.cwd}/actor.pth'
                #pdb.set_trace()
                act_save_path = f'{self.distance_path}/best_actor.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
                # pdb.set_trace()
                print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print
            # pdb.set_trace()

        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        return if_reach_goal, if_save

    def evaluate_and_save_injection_modeling(self, act, steps, r_exp, log_tuple, cri_scheduler, act_scheduler) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            self.r_max = r_exp  # update max reward (episode return)
            # act_save_path = f'{self.cwd}/actor.pth'
            #pdb.set_trace()
            act_save_path = f'{self.distance_path}/best_actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            # pdb.set_trace()
            print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        return if_reach_goal, if_save

    def evaluate_and_save_injection_modeling_rmse(self, act, steps, r_exp, rmse) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        #self.writer.add_scalar('RMSE', rmse, self.total_step)
        self.writer.add_scalar('RMSE', rmse[0], self.total_step-400)
        self.writer.add_scalar('RMSE', rmse[1], self.total_step - 200)
        self.writer.add_scalar('RMSE', rmse[2], self.total_step)
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            self.r_max = r_exp  # update max reward (episode return)
            # act_save_path = f'{self.cwd}/actor.pth'
            #pdb.set_trace()
            act_save_path = f'{self.distance_path}/best_actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            # pdb.set_trace()
            print(f"{self.agent_id:<10000}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        return if_reach_goal, if_save

    def evaluate_and_save_SISO_CSTR(self, act, steps, r_exp, log_tuple, cri_scheduler, act_scheduler) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            self.r_max = r_exp  # update max reward (episode return)
            # act_save_path = f'{self.cwd}/actor.pth'
            #pdb.set_trace()
            act_save_path = f'{self.distance_path}/best_actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            # pdb.set_trace()
            print(f"{self.agent_id:<10}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        return if_reach_goal, if_save


    def evaluate_and_save_SISO_CSTR_rmse(self, act, steps, r_exp, rmse) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps
        """"""

        self.writer.add_scalar('episodemeanReward', r_exp, self.total_step)
        #self.writer.add_scalar('RMSE', rmse, self.total_step)
        for item in range(10):
            #pdb.set_trace()
            self.writer.add_scalar('RMSE', rmse[item], self.total_step - 300*(9-item))
        '''save the policy network'''
        if_save = r_exp > self.r_max
        if if_save:  # save checkpoint with highest episode return
            self.r_max = r_exp  # update max reward (episode return)
            # act_save_path = f'{self.cwd}/actor.pth'
            #pdb.set_trace()
            act_save_path = f'{self.distance_path}/best_actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            # pdb.set_trace()
            print(f"{self.agent_id:<10}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        '''print some information to Terminal'''
        # pdb.set_trace()
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        return if_reach_goal, if_save




    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            #pdb.set_trace()
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    pdb.set_trace()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,),dtype=torch.float32, device=device)
        #pdb.set_trace()
        a_tensor = act(s_tensor)
        #pdb.set_trace()
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        #pdb.set_trace()
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    #pdb.set_trace()
    #episode_return = getattr(env, 'episode_return', episode_return)
    #pdb.set_trace()
    return episode_return, episode_step

def get_episode_return_and_step_state(env, act, device) -> (float, int,list):
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_step
    if_discrete = env.if_discrete
    episode_state=[]
    state = env.reset()
    #pdb.set_trace()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,),dtype=torch.float32, device=device)
        #pdb.set_trace()
        a_tensor = act(s_tensor)
        #pdb.set_trace()
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        #pdb.set_trace()
        state, reward, done, _ = env.step(action)
        episode_state.append(state)
        episode_return += reward
        if done:
            break
    #pdb.set_trace()
    #episode_return = getattr(env, 'episode_return', episode_return)
    #pdb.set_trace()
    return episode_return, episode_step,episode_state


def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return')
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
