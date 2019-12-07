import gym
import numpy as np
from matplotlib import pyplot as plt
from itertools import count
import torch.optim as optim
import random
import torch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import Transition, ReplayMemory, plot_rewards
# initialize one and start from that in the forward loop

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent(nn.Module):

    def __init__(self, q_models, target_model, hyperbolic, k, gamma, model_params, replay_buffer_size, batch_size,
                 inp_dim):
        super(Agent, self).__init__()
        if hyperbolic:
            self.q_models = torch.nn.ModuleList(q_models)
            self.target_models = torch.nn.ModuleList(target_model)
        else:
            self.q_models = q_models
            self.target_models = target_model
        self.optimizer = optim.RMSprop(self.q_models.parameters(), lr=1e-3)
        self.hyperbolic = hyperbolic
        self.n_actions = model_params.act_space
        self.k = k
        self.gamma = gamma
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.inp_dim = inp_dim

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    @staticmethod
    def get_hyperbolic_train_coeffs(k, num_models):
        coeffs = []
        gamma_intervals = np.linspace(0, 1, num_models + 1)
        for i in range(num_models):
            coeffs.append((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** (1 / (k - 1)))
        return torch.tensor(coeffs)

    def get_action(self, state_batch, epsilon=0.05):
        model_outputs = []
        take_random_action = random.random()
        if take_random_action > epsilon:
            return random.randrange(self.n_actions)
        elif self.hyperbolic:
            state_batch = torch.tensor(state_batch, dtype=torch.float32).view(-1, self.inp_dim)
            for ind, mdl in enumerate(self.q_models):
                model_outputs.append(mdl(state_batch))
            coeff = self.get_hyperbolic_train_coeffs(self.k, len(self.q_models))
            model_outputs = torch.cat(model_outputs, 1).reshape(-1, len(self.q_models))
            model_outputs = (model_outputs * coeff).sum(dim=1)
            return torch.argmax(model_outputs).item()

    def get_state_act_vals(self, state_batch, action_batch=None):
        if self.hyperbolic:
            model_outputs = []
            for ind, mdl in enumerate(self.q_models):
                model_outputs.append(mdl(state_batch).gather(1, action_batch))
            model_outputs = torch.cat(model_outputs, 1).reshape(-1, len(self.q_models))
            coeffs = self.get_hyperbolic_train_coeffs(self.k, len(self.q_models))
            model_outputs = model_outputs * coeffs
            return model_outputs.sum(dim=1).reshape(-1, 1)
        else:
            model_output = self.q_models(state_batch).gather(1, action_batch)
            return model_output

    def get_max_next_state_vals(self, non_final_mask, non_final_next_states):
        if self.hyperbolic:
            target_outptus = []
            for ind, mdl in enumerate(self.target_models):
                next_state_values = torch.zeros(self.batch_size)
                next_state_values[non_final_mask] = mdl(non_final_next_states).max(1)[0].detach()
                target_outptus.append(next_state_values)
            target_outptus = torch.cat(target_outptus, 0).reshape(-1, len(self.target_models))
            coeffs = self.get_hyperbolic_train_coeffs(self.k, len(self.q_models))
            target_outptus = target_outptus * coeffs
            return target_outptus.sum(dim=1).reshape(-1, 1)

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.get_state_act_vals(state_batch, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self.get_max_next_state_vals(non_final_mask, non_final_next_states)
        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = self.gamma * next_state_values + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_models.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)


def initialize_env(env_name):
    env_name = env_name
    env = gym.make(env_name)
    env.reset()
    if args.env != "CartPole-v0":
        print("No implementation for environment " + args.env + ". Exiting...")
        exit(1)
    return env


def initialize_model(model_params, train_hyperbolic, k, gamma, num_models, replay_buffer, batch_size):
    target_list = []
    q_val_list = []
    if train_hyperbolic:
        for _ in range(num_models):
            q_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
            target_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
            target_model.load_state_dict(q_model.state_dict())
            q_val_list.append(q_model)
            target_list.append(target_model)
        return Agent(q_val_list, target_list, train_hyperbolic, k, gamma, model_params, replay_buffer, batch_size,
                     model_params.inp_dim)
    else:
        q_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
        target_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
        target_model.load_state_dict(q_model.state_dict())
        return Agent(q_model, target_model, train_hyperbolic, k, gamma, model_params, replay_buffer, batch_size,
                     model_params.inp_dim)


def train(num_episodes, glie_a, agent):
    cumulative_rewards = []
    for ep in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = False
        eps = glie_a / (glie_a + ep)
        cum_reward = 0
        while not done:
            # Select and perform an action
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward

            # Task 1: TODO: Update the Q-values
            # agent.single_update(state, action, next_state, reward, done)
            agent.store_transition(state, action, next_state, reward, done)
            # agent.update_estimator()
            # Task 2: TODO: Store transition and batch-update Q-values
            # Task 4: Update the DQN
            agent.update_network()

            # Move to the next state
            state = next_state
        cumulative_rewards.append(cum_reward)
        if ep == num_episodes - 1:
            plot_rewards(cumulative_rewards, save_fig=True)
        print("Cumulative reward on episode", ep, " ", cum_reward)


def initialize_model_and_training_params(args, env):
    glie_a = args.glie_a
    num_episodes = args.num_episodes
    TARGET_UPDATE = args.target_update
    hidden = args.hidden_dim
    gamma = args.gamma
    replay_buffer_size = args.replay_buffer
    batch_size = args.batch_size
    n_actions = env.action_space.n
    state_space_dim = env.observation_space.shape[0]
    num_models = args.num_models if args.train_hyperbolic else 1
    Model_Info = namedtuple('Model_Info',
                            ('inp_dim', 'act_space', 'hidden_size'))
    Train_Info = namedtuple('Train_Info',
                            ('glie_a', 'num_episodes', 'batch_size', 'replay_buffer_size', 'gamma', 'num_models'))
    model_info = Model_Info(state_space_dim, n_actions, hidden)
    train_info = Train_Info(glie_a, num_episodes, batch_size, replay_buffer_size, gamma, num_models)
    return model_info, train_info


def parse_arguments():
    parser = argparse.ArgumentParser()
    # continuous cart-pole is most likely suited only for actor-critic methods
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--normalize_rewards", default=False, action='store_true', help="use zero mean/unit variance"
                                                                                        "normalization")
    parser.add_argument("--baseline", type=int, default=0, help="Baseline to 0 (default=0 - no baseline)")
    parser.add_argument("--sigma_type", type=str, default="constant", help="Learn sigma as a model parameter")
    parser.add_argument("--hidden_dim", type=int, default=12, help="Model hidden size required")
    parser.add_argument("--glie_a", type=int, default=200, help="Parameter for lowering the exploration of the "
                                                                "model during the training")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of episodes to use for training"
                                                                       "the model")
    parser.add_argument("--gamma", type=float, default=0.98, help="Learn sigma as a model parameter")
    parser.add_argument("--k", type=float, default=0.1, help="Learn sigma as a model parameter")
    parser.add_argument("--batch_size", type=int, default=32, help="Learn sigma as a model parameter")
    parser.add_argument("--replay_buffer", type=int, default=50000, help="Learn sigma as a model parameter")
    parser.add_argument("--num_models", type=int, default=50, help="Number of models to be used for ")
    parser.add_argument("--train_hyperbolic", default=False, action="store_true",
                        help="Train using hyperbolic discounting")
    parser.add_argument("--target_update", default=False, help="Train using hyperbolic discounting")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    env = initialize_env(args.env)
    model_params, training_params = initialize_model_and_training_params(args, env)
    agent = initialize_model(model_params, args.train_hyperbolic, args.k, args.gamma, args.num_models,
                             args.replay_buffer, args.batch_size)
    train(args.num_episodes, args.glie_a, agent)
