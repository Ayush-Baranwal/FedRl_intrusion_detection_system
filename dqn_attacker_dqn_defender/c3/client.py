from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import flwr as fl

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym_idsgame.envs import IdsGameEnv
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn_config import DQNConfig
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn import DQNAgent


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IntrusionClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        random_seed = 0
        dqn_config = DQNConfig(input_dim=242, defender_output_dim=242, attacker_output_dim=242, hidden_dim=32, replay_memory_size=10000,
                               num_hidden_layers=1,
                               replay_start_size=1000, batch_size=32, target_network_update_freq=1000,
                               gpu=True, tensorboard=False,
                               loss_fn="Huber", optimizer="Adam", lr_exp_decay=True, lr_decay_rate=0.9999)
        q_agent_config = QAgentConfig(gamma=0.999, alpha=0.00003, epsilon=1, render=False, eval_sleep=0.9,
                                      min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                      epsilon_decay=0.9999, video=False, eval_log_frequency=1,
                                      video_fps=5, video_dir="./results/videos", num_episodes=1000,
                                      eval_render=False, gifs=True, gif_dir="./results/gifs",
                                      eval_frequency=100, attacker=True, defender=True, video_frequency=101,
                                      save_dir="./results/data", dqn_config=dqn_config,
                                      checkpoint_freq=1000)
        env_name = "idsgame-v4"
        env = gym.make(env_name, save_dir="./results/data")
        self.attacker_agent = DQNAgent(env, q_agent_config)

    def get_parameters(self, config):

        param_list = dict()
        param_list["attacker_state"] = [val.cpu().numpy(
        ) for _, val in self.attacker_agent.attacker_q_network.state_dict().items()]
        param_list["defender_state"] = [val.cpu().numpy(
        ) for _, val in self.attacker_agent.defender_q_network.state_dict().items()]
        return param_list["defender_state"]
        # return [val.cpu().numpy() for _, val in attacker_agent.state_dict().items()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict_attacker = zip(self.attacker_agent.defender_q_network.state_dict(
        ).keys(), parameters)
        # params_dict_defender = zip(self.attacker_agent.defender_q_network.state_dict(
        # ).keys(), parameters["defender_state"])

        state_dict_attacker = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict_attacker})
        # state_dict_defender = OrderedDict(
        #     {k: torch.tensor(v) for k, v in params_dict_defender})
        self.attacker_agent.defender_q_network.load_state_dict(
            state_dict_attacker, strict=True)
        # self.attacker_agent.defender_target_network.load_state_dict(
        #     state_dict_defender)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.attacker_agent.train()
        # attacker_agent.train_step(parameters)
        return self.get_parameters(config={}), 1, {"accuracy": 0.95}

    # def evaluate(
    #     self, parameters: List[np.ndarray], config: Dict[str, str]
    # ) -> Tuple[float, int, Dict]:
    #     self.set_parameters(parameters)

    #     self.attacker_agent.eval(100)
    #     return self.attacker_agent.log_metrics()
    #     # loss, accuracy = attacker_agent.evaluate(x_test, y_test)
    #     # return loss, len(x_test), {"accuracy": accuracy}

    #     # self.set_parameters(parameters)
    #     # loss, accuracy = test(net, testloader)
    #     # return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    # Load model and data

    # Start client
    client = IntrusionClient()
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
