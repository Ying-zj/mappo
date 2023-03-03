import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']  # 实验参数
        self.envs = config['envs']  # 训练环境
        self.eval_envs = config['eval_envs']  # 测试环境
        self.device = config['device']  # 设备，cpu or gpu
        self.num_agents = config['num_agents']  # 智能体数量
        if config.__contains__("render_envs"):  # 如果包括render环境，则添加属性
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name  # 环境名称
        self.algorithm_name = self.all_args.algorithm_name  # 算法名称
        self.experiment_name = self.all_args.experiment_name  # 实验名称
        self.use_centralized_V = self.all_args.use_centralized_V  # 是否使用中心化的V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state  # 是否使用观测来代替全局状态
        self.num_env_steps = self.all_args.num_env_steps  # 环境最大步数
        self.episode_length = self.all_args.episode_length  # episode长度
        self.n_rollout_threads = self.all_args.n_rollout_threads  # 训练所需线程
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 测试所需线程
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads  # render所需线程，干嘛用的？
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay  # 不知
        self.hidden_size = self.all_args.hidden_size  # 隐层大小
        self.use_wandb = self.all_args.use_wandb  # 是否使用wandb
        self.use_render = self.all_args.use_render  # 是否使用render
        self.recurrent_N = self.all_args.recurrent_N  # 周期n，不知

        # interval
        self.save_interval = self.all_args.save_interval  # 保存间隔
        self.use_eval = self.all_args.use_eval  # 是否测试
        self.eval_interval = self.all_args.eval_interval  # 测试间隔
        self.log_interval = self.all_args.log_interval  # 存入日志间隔

        # dir
        self.model_dir = self.all_args.model_dir  # 模型保存路径

        if self.use_wandb:  # 使用wandb
            self.save_dir = str(wandb.run.dir)  # 保存的路径
            self.run_dir = str(wandb.run.dir)  # 运行的路径
        else:
            self.run_dir = config["run_dir"]  # 如果不使用wandb就使用配置中的run_dir
            self.log_dir = str(self.run_dir / 'logs')  # 日志路径
            if not os.path.exists(self.log_dir):  # 创建日志路径
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)  # 一个将条目直接写入日志供tensorboard使用
            self.save_dir = str(self.run_dir / 'models')  # 模型保存路径
            if not os.path.exists(self.save_dir):  # 创建保存路径
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]  # 共享观测空间

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)  # 创建一个rmappo的策略网络

        if self.model_dir is not None:  # 存储模型
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)  # rmappo训练器
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])  # 建立共享rb

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()  # 将training改为False
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))  # critic做一次前向传播
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
