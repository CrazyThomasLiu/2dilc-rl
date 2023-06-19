import os
import time
import shutil
import torch
import numpy as np
import numpy.random as rd
from env_function import build_env
from replay import ReplayBuffer
from evaluator import Evaluator
import pdb
import yaml
import time


import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")


class Arguments:
    def __init__(self, if_on_policy=False):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-10000

        self.if_on_policy = if_on_policy
        if self.if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 3  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU

    def init_before_training(self, if_main):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)
        #pdb.set_trace()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

        '''env'''
        if self.env is None:
            raise RuntimeError(f'\n| Why env=None? For example:'
                               f'\n| args.env = XxxEnv()'
                               f'\n| args.env = str(env_name)'
                               f'\n| args.env = build_env(env_name), from elegantrl.env import build_env')
        if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

        '''agent'''
        if self.agent is None:
            raise RuntimeError(f'\n| Why agent=None? Assignment `args.agent = AgentXXX` please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError(f"\n| why hasattr(self.agent, 'init') == False"
                               f'\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`.')
        if self.agent.if_on_policy != self.if_on_policy:
            raise RuntimeError(f'\n| Why bool `if_on_policy` is not consistent?'
                               f'\n| self.if_on_policy: {self.if_on_policy}'
                               f'\n| self.agent.if_on_policy: {self.agent.if_on_policy}')

        '''cwd'''
        """path for the saving files"""
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            current_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_path)
            current_dir = f'{current_dir}/runs_{agent_name}_{env_name}_nominal'
            number = 1
            self.cwd = os.path.join(current_dir, str(number))
            while os.path.exists(self.cwd):
                number += 1
                self.cwd = os.path.join(current_dir, str(number))
            os.makedirs(self.cwd)

        if if_main:
            # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=False)
    #pdb.set_trace()
    env = build_env(args.env, if_print=False)

    '''init: Agent'''
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae, args.env_num)
    #pdb.set_trace()
    #agent.save_or_load_agent(args.cwd, if_save=False)

    "Save all data into the yaml"
    save_args_path = os.path.join(args.cwd, 'args.yaml')
    fp = open(save_args_path, 'w')
    fp.write(yaml.dump(args.__dict__))

    '''init Evaluator'''
    #pdb.set_trace()
    eval_env = build_env(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_gap, args.eval_times1, args.eval_times2,args.cwd)
    #evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae)
        #buffer.save_or_load_history(args.cwd, if_save=False)

    #pdb.set_trace()
    agent.save_or_load_agent_implementation(args.cwd, if_save=False)
    #buffer.save_or_load_history_implementation(args.cwd, if_save=False) if not agent.if_on_policy else None
    "buffer load in the next line"
    env.save_or_load_history_implementation(args.cwd, if_save=False)
    #pdb.set_trace()
    'test for the training time'

    """start training"""
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''choose update_buffer()'''
    if agent.if_on_policy:
        assert isinstance(buffer, list)

        def update_buffer(_trajectory):
            _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
            ten_state = torch.as_tensor(_trajectory[0])
            ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
            ten_action = torch.as_tensor(_trajectory[3])
            ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

            buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

            _steps = ten_reward.shape[0]
            _r_exp = ten_reward.mean()
            return _steps, _r_exp
    else:
        assert isinstance(buffer, ReplayBuffer)

        def update_buffer(_trajectory_list):
            #pdb.set_trace()
            _steps = 0
            _r_exp = 0
            for _trajectory in _trajectory_list:
                #pdb.set_trace()
                """
                ten_state_compare=torch.empty(10000)

                for item in _trajectory:
                    pdb.set_trace()
                    a=torch.as_tensor(np.array(item[0]), dtype=torch.float32)
                    b=torch.cat((a,a),0)
                    pdb.set_trace()
                pdb.set_trace()
                #ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
                #for item in _trajectory:
                #    ten_state_compare=torch.cat((ten_state_compare,torch.as_tensor(item[0], dtype=torch.float32)),1)

                pdb.set_trace()
                """
                #pdb.set_trace()
                ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
                ary_other = torch.as_tensor([item[1] for item in _trajectory])
                #pdb.set_trace()
                ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ten_reward
                ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
                #pdb.set_trace()

                buffer.extend_buffer(ten_state, ary_other)

                _steps += ten_state.shape[0]
                _r_exp += ary_other[:, 0].mean()  # other = (reward, mask, action)
            return _steps, _r_exp
    '''init ReplayBuffer after training start'''
    agent.states = [env.reset(), ] #????? 导致 batch 增加
    if not agent.if_on_policy:
        #pdb.set_trace()
        if_load = buffer.save_or_load_history_implementation(cwd, if_save=False)

        if not if_load:
            #pdb.set_trace()
            trajectory = explore_before_training(env, target_step)
            #pdb.set_trace()
            trajectory = [trajectory, ]
            steps, r_exp = update_buffer(trajectory)
            evaluator.total_step += steps

    '''start training loop'''
    if_train = True
    while if_train:
        starttime = time.time()
        with torch.no_grad():
            #pdb.set_trace()

            trajectory = agent.explore_env(env, target_step)
            #pdb.set_trace()
            steps, r_exp = update_buffer(trajectory)
        #pdb.set_trace()

        #pdb.set_trace()
        #starttime = time.time()
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        #endtime = time.time()
        #duringtime = endtime - starttime
        #print(duringtime)
        with torch.no_grad():
            #pdb.set_trace()
            #temp = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            #temp = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple , agent.cri_scheduler, agent.act_scheduler)
            #temp = evaluator.evaluate_and_save_liu(agent.act, steps, r_exp, logging_tuple , agent.cri_scheduler, agent.act_scheduler)
            #temp = evaluator.evaluate_and_save_evalenv(agent.act, steps, r_exp, logging_tuple, agent.cri_scheduler,agent.act_scheduler)
            #temp = evaluator.evaluate_and_save_injection_modeling(agent.act, steps, r_exp, logging_tuple, agent.cri_scheduler,
            #                                           agent.act_scheduler)

            temp = evaluator.evaluate_and_save_SISO_CSTR(agent.act, steps, r_exp, logging_tuple,
                                                                  agent.cri_scheduler,
                                                                  agent.act_scheduler)
            #########################################################################
            #########################################################################
            if_reach_goal, if_save = temp
            #pdb.set_trace()
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close()
    """
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)
    """



    #agent.save_or_load_agent_implementation(cwd, if_save=True)
    agent.save_or_load_agent_implementation(cwd, if_save=True)
    buffer.save_or_load_history_implementation(cwd, if_save=True) if not agent.if_on_policy else None
    env.save_or_load_history_implementation(cwd, if_save=True)
    evaluator.save_or_load_recoder(if_save=True)


def explore_before_training(env, target_step):  # for off-policy only
    trajectory = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()  ###???? batch increase
    step = 0
    while True:
        if if_discrete:
            action = rd.randint(action_dim)  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, action)
        else:
            """Jianan Liu
            """
            #TODO: why [-1 1]
            action = rd.uniform(-1, 1, size=action_dim)
            #action = rd.uniform(-0.1, 0.1, size=action_dim)
            #pdb.set_trace()
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, *action)

        trajectory.append((state, other))
        state = env.reset() if done else next_s
        #pdb.set_trace()
        step += 1
        if done and step > target_step: # ????? one episode?
            #pdb.set_trace()
            break
    return trajectory


def explore_before_training_vec_env(env, target_step) -> list:  # for off-policy only
    # plan to be elegant: merge this function to explore_before_training()
    assert hasattr(env, 'env_num')
    env_num = env.env_num

    trajectory_list = [list() for _ in range(env_num)]

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    states = env.reset()
    step = 0
    while True:
        if if_discrete:
            actions = rd.randint(action_dim, size=env_num)
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, actions[env_i])))
                next_states.append(next_s)
        else:
            actions = rd.uniform(-1, 1, size=(env_num, action_dim))
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, *actions[env_i])))
                next_states.append(next_s)
        states = next_states

        step += 1
        if step > target_step:
            break
    return trajectory_list
