import os
import torch
import numpy as np
import numpy.random as rd

from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from net import ActorSAC
from net import Critic, CriticTwin
import math
import pdb


class AgentBase:
    def __init__(self):
        self.states = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.explore_rate = 1.0
        self.explore_noise = None
        self.traj_list = None  # trajectory_list
        # self.amp_scale = None  # automatic mixed precision
        """ Jianan Liu
        """
        self.cri_scheduler=None
        self.act_scheduler=None
        self.alpha_scheduler = None
        self.clip_grad_norm = 4.0
        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4,
             if_per_or_gae=False, env_num=1, agent_id=0):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `float learning_rate` learning rate of optimizer
        `bool if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `int env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `int agent_id` if the visible_gpu is '1,9,10000,50000', agent_id=1 means (1,9,50000,10000)[agent_id] == 9
        """
        self.action_dim = action_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{agent_id}" if (torch.cuda.is_available() and (agent_id >= 0)) else "cpu")

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        """
        jianan Liu scheduler
        """
        #pdb.set_trace()
        #self.cri_scheduler = torch.optim.lr_scheduler.StepLR(self.cri_optim, step_size=100, gamma=0.94)
        #self.act_scheduler=torch.optim.lr_scheduler.StepLR(self.act_optim, step_size=100, gamma=0.94)
        #self.cri_scheduler=torch.optim.lr_scheduler.StepLR(self.cri_optim,step_size=100,gamma=0.97)
        #self.act_scheduler=torch.optim.lr_scheduler.StepLR(self.act_optim, step_size=100, gamma=0.97)
        self.cri_scheduler=torch.optim.lr_scheduler.StepLR(self.cri_optim,step_size=100,gamma=1)
        self.act_scheduler=torch.optim.lr_scheduler.StepLR(self.act_optim, step_size=100, gamma=1)
        #self.cri_scheduler=torch.optim.lr_scheduler.StepLR(self.cri_optim,step_size=1800,gamma=0.99)
        #self.act_scheduler=torch.optim.lr_scheduler.StepLR(self.act_optim, step_size=1800, gamma=0.99)
        del self.ClassCri, self.ClassAct

        if env_num > 1:  # VectorEnv
            self.explore_env = self.explore_vec_env
        else:
            self.explore_env = self.explore_one_env

    def select_actions(self, states) -> np.ndarray:
        """Select continuous actions for exploration

        `array states` states.shape==(batch_size, state_dim, )
        return `array actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        #pdb.set_trace()
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = (actions + torch.randn_like(actions) * self.explore_noise).clamp(-1, 1)
        return actions.detach().cpu().numpy()

    def explore_one_env(self, env, target_step, time_length=200):
        """actor explores in one env, then returns the traj (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        traj = list()
        #pdb.set_trace()
        state = self.states[0] # ??????define the initial  state
        'control performance RMSE'
        rmse = []
        sum = 0.0
        error = 0.0
        #pdb.set_trace()
        for item in range(target_step):
            action = self.select_actions((state,))[0]
            #print(action)
            next_s, reward, done, _ = env.step(action)
            #pdb.set_trace()
            traj.append((state, (reward, done, *action)))

            state = env.reset() if done else next_s

            error = (next_s[6] - next_s[9]) ** 2
            sum += error
            if ( item + 1) % time_length == 0:
                tem = math.sqrt(sum / time_length)
                rmse.append(tem)
                sum = 0.0
        self.states[0] = state
        #pdb.set_trace()
        traj_list = [traj, ]
        return traj_list, rmse  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        """actor explores in VectorEnv, then returns the trajectory (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        env_num = len(self.traj_list)
        states = self.states

        traj_list = [list() for _ in range(env_num)]
        for _ in range(target_step):
            actions = self.select_actions(states)
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_state, reward, done = s_r_d_list[env_i]
                traj_list[env_i].append(
                    (states[env_i], (reward, done, *actions[env_i]))
                )
                next_states.append(next_state)
            states = next_states

        self.states = states
        return traj_list  # (traj_env_0, ..., traj_env_i)

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer.
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        `float repeat_times` the times of sample batch = int(target_step * repeat_times) in off-policy
        `float soft_update_tau` target_net = target_net * (1-tau) + current_net * tau
        `return tuple` training logging. tuple = (float, float, ...)
        """

    def optim_update_new_scheduler(self, optimizer, objective, params,scheduler):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()
        scheduler.step()

    def optim_update_new(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    #@staticmethod
    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=10000.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()
    """
    Jianan Liu  ##########################10000

    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()

        optimizer.step()

    """
    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        `nn.Module target_net` target network update via a current network, it is more stable
        `nn.Module current_net` current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.

        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        #pdb.set_trace()
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None
###################################################################
    # added from the AgentZoo/ElegantRL-Isaac/elegantrl/agent.py
    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        pdb.set_trace()
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_per=False, env_num=1, agent_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_per, env_num, agent_id)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)
        #TODO SmoothL1Loss is squared not suitable for the MIMO-CSTR
        if if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_use_per else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_use_per else 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        #pdb.set_trace()
        #TODO if traning is finished, then set the explore-rate to 0
        #if rd.rand() < self.explore_rate:  # epsilon-greedy
        #    actions = self.act.get_action(states)
        #else:
        #    actions = self.act(states)
        actions = self.act(states)
        return actions.detach().cpu().numpy()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()
            #pdb.set_trace()
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            pdb.set_trace()
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2).detach()
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)

            self.soft_update(self.act_target, self.act, soft_update_tau)
        return self.obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            #pdb.set_trace()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q2).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self):
        super().__init__()
        self.if_use_act_target = True
        self.if_use_cri_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        #pdb.set_trace()
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()
            #pdb.set_trace()
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            #pdb.set_trace()
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * obj_critic.item()  # for reliable_lambda
            """jianan liu"""
            #pdb.set_trace()
            #self.optim_update(self.cri_optim, obj_critic)
            #self.optim_update_new(self.cri_optim, obj_critic, self.cri.parameters())
            self.optim_update_new_scheduler(self.cri_optim, obj_critic, self.cri.parameters(),self.cri_scheduler)
            #pdb.set_trace()
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
            #pdb.set_trace()
            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            """jianan liu"""
            #self.optim_update(self.alpha_optim, obj_alpha)
            self.optim_update_new(self.alpha_optim, obj_alpha, self.alpha_log)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()
            #pdb.set_trace()
            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + logprob * alpha).mean()
                """jianan liu"""
                #self.optim_update(self.act_optim, obj_actor)
                #self.optim_update_new(self.act_optim, obj_actor, self.act.parameters())
                self.optim_update_new_scheduler(self.act_optim, obj_actor, self.act.parameters(),self.act_scheduler)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        #pdb.set_trace()
        return self.obj_critic, obj_actor.item(), alpha.item()
    def save_or_load_agent_implementation(self, cwd, if_save):
        """save or load the training files for agent from disk.

        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim),
                         ('alpha_optim', self.alpha_optim),]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        #pdb.set_trace()
        if if_save:
            #pdb.set_trace()
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
            'self.alpha_log is not network, which do not have the stata_dict()'
            tem_alpha_log=self.alpha_log.detach().cpu().numpy()
            save_path = f"{cwd}/alpha_log.npz"
            np.savez_compressed(save_path, alpha_log=tem_alpha_log,)
            #pdb.set_trace()
            print(f"| agent save in: {save_path}")

        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

            save_path = f"{cwd}/alpha_log.npz"
            if os.path.isfile(save_path):
                agent_dict = np.load(save_path)
                tem_alpha_log = agent_dict['alpha_log']
                #pdb.set_trace()
                with torch.no_grad():
                    self.alpha_log[0].fill_(float(tem_alpha_log))
                #tem_alpha_log = torch.as_tensor(tem_alpha_log, dtype=torch.float32, requires_grad=True,device=self.device)
                #pdb.set_trace()
            print(f"| agent load: {save_path}")
