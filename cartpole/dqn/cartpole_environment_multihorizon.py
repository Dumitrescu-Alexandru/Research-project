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

TARGET_UPDATE = 100


# initialize one and start from that in the forward loop

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12, no_models=100):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim * no_models)
        self.no_models = no_models

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent(nn.Module):

    def __init__(self, q_models, target_model, hyperbolic, k, gamma, model_params, replay_buffer_size, batch_size,
                 inp_dim, lr, no_models, act_space, hidden_size, loss_type, target_update=False):
        super(Agent, self).__init__()
        if hyperbolic:
            self.q_models = DQN(state_space_dim=inp_dim, action_space_dim=act_space, hidden=hidden_size,
                                no_models=no_models)
            self.target_models = DQN(state_space_dim=inp_dim, action_space_dim=act_space, hidden=hidden_size,
                                     no_models=no_models)
            self.target_models.load_state_dict(self.q_models.state_dict())
            self.target_models.eval()
        else:
            self.q_models = q_models
        self.optimizer = optim.RMSprop(self.q_models.parameters(), lr=lr)
        self.hyperbolic = hyperbolic
        self.n_actions = model_params.act_space
        self.k = k
        # self.gammas = torch.tensor(np.linspace(0, 1, self.q_models.no_models + 1), dtype=torch.float)[1:]
        self.gammas = np.sort(np.random.uniform(0, 1, self.q_models.no_models + 1))
        self.gammas = np.append(self.gammas, 0.98)
        self.gammas = torch.tensor(np.sort(self.gammas))
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.inp_dim = inp_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_models.to(self.device)
        self.q_models.to(self.device)
        self.gammas = self.gammas.to(self.device)
        self.loss_type = loss_type
        self.criterion = nn.MSELoss()
        self.use_target_network = target_update

    def update_network(self, updates=1):
        for _ in range(updates):
            loss = self._do_network_update()
        return loss

    def get_hyperbolic_train_coeffs(self, k, num_models):
        coeffs = []
        for i in range(1, num_models + 1):
            coeffs.append(
                ((self.gammas[i + 1] - self.gammas[i]) * (1 / k) * self.gammas[i] ** ((1 / k) - 1)))
        return torch.tensor(coeffs).to(self.device) / sum(coeffs)

    def get_action(self, state_batch, epsilon=0.05, get_among_last=False):
        # epsilon gets smaller as time goes by. 
        # (glie_a/(glie_a + eps)) with eps in range(0, no_episodes)
        take_random_action = random.random()
        if take_random_action < epsilon:
            return random.randrange(self.n_actions)
        elif get_among_last:
            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device).view(-1,
                                                                                                  self.inp_dim)
            model_outputs = self.q_models(state_batch).reshape(2, self.q_models.no_models)
            return torch.argmax(model_outputs[:, -10].view(-1)).item()
            model_outputs = model_outputs * self.get_hyperbolic_train_coeffs(self.k, self.q_models.no_models)
            actions = torch.argmax(torch.sum(model_outputs, dim=1))
            return actions.item()
        elif self.hyperbolic:
            with torch.no_grad():
                state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device).view(-1, self.inp_dim)
                model_outputs = self.q_models(state_batch.double()).reshape(-1, 2)
                coeffs = self.get_hyperbolic_train_coeffs(self.k, self.q_models.no_models).reshape(-1, 1)
                model_outputs = model_outputs * coeffs
                actions = torch.argmax(torch.sum(model_outputs, dim=0))
            return actions.item()

    def get_state_act_vals(self, state_batch, action_batch=None):
        if self.hyperbolic:
            action_batch = action_batch.repeat(1, self.q_models.no_models).reshape(-1, 1)
            model_outputs = self.q_models(state_batch.to(self.device).double())
            model_outputs = model_outputs.reshape(-1, self.n_actions)
            model_outputs = model_outputs.gather(1, action_batch)
            # .reshape(self.q_models.no_models * state_batch.shape[0],
            #          2).gather(1, action_batch.reshape(-1))
            return model_outputs
        else:
            model_output = self.q_models(state_batch).gather(1, action_batch)
            return model_output

    def get_max_next_state_vals(self, non_final_mask, non_final_next_states):
        if self.hyperbolic:
            with torch.no_grad():
                next_state_values = torch.zeros(self.batch_size).to(self.device)
                # doing it like this, the model_no will come first and then the batch_no (b1m1, b1m2, b1m3..., b2m1,
                # ...b10m1, b10m2...
                # if False in non_final_mask:
                #     print(non_final_mask)
                #     print(len(non_final_next_states))
                non_final_mask = non_final_mask.reshape(-1, 1).repeat(1, self.q_models.no_models).view(-1)
                # if False in non_final_mask:
                #     print([nf for nf in non_final_mask])
                next_state_values = next_state_values.view(-1, 1).repeat(1, self.q_models.no_models).view(-1)
                if self.use_target_network:
                    # [b1m1o1, b1m1o2], -> max -> [b1m1]
                    # [b1m2o1, b1m2o2],           [b1m2]
                    # [b1m3o1, b1m3o3],           [b1m3]
                    # ...                         ...
                    #
                    next_state_values[non_final_mask] = \
                        self.target_models(non_final_next_states.to(self.device)).reshape(-1, self.n_actions).max(1)[0]
                    # if False in non_final_mask:
                    #     print("first", self.target_models(non_final_next_states.to(self.device)))
                    #     print("after reshaping", self.target_models(non_final_next_states.to(self.device)).reshape(-1, self.n_actions))
                    #     print(self.target_models(non_final_next_states.to(self.device)).shape)
                    #     print("next_state_values", next_state_values)
                else:
                    next_state_values[non_final_mask] = \
                        self.q_models(non_final_next_states.to(self.device)).reshape(-1, self.n_actions).max(1)[0]
                target_outptus = next_state_values
                return target_outptus * self.gammas[2:].repeat(self.batch_size)

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_state) if nonfinal]
        non_final_next_states = torch.stack(non_final_next_states).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        state_action_values = self.get_state_act_vals(state_batch, action_batch).view(-1)
        next_state_values = self.get_max_next_state_vals(non_final_mask, non_final_next_states)
        # this should be perfect
        expected_state_action_values = next_state_values + \
                                       reward_batch.view(-1, 1).repeat(1, self.q_models.no_models).view(-1)
        # print(reward_batch.view(-1, 1).repeat(1, self.q_models.no_models).view(-1).shape)
        if self.loss_type == "weighted_loss":
            loss = (state_action_values - expected_state_action_values) ** 2
            hyp_coef = self.get_hyperbolic_train_coeffs(self.k, self.q_models.no_models).repeat(self.batch_size)
            loss = (loss.reshape(-1).view(-1) * hyp_coef).view(-1)
            loss = torch.mean(loss)
        elif self.loss_type == "separate_summarized_loss":
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values).double()
            # loss = (state_action_values - expected_state_action_values) ** 2
            # loss = torch.sum(loss)
        elif self.loss_type == "one_output_loss":
            hyp_coef = self.get_hyperbolic_train_coeffs(self.k, self.q_models.no_models)
            state_action_values = state_action_values.reshape(self.batch_size, -1) * hyp_coef
            state_action_values = torch.sum(state_action_values, dim=1)
            expected_state_action_values = expected_state_action_values.reshape(self.batch_size, -1) * hyp_coef
            expected_state_action_values = torch.sum(expected_state_action_values, dim=1)
            loss = self.criterion(state_action_values, expected_state_action_values)

        loss_item = loss.item()
        # print(hyp_coef.repeat(self.batch_size).shape)
        # print(loss.shape)
        # loss = (state_action_values - expected_state_action_values) ** 2 * self.get_hyperbolic_train_coeffs(self.k,
        #                                                                                                     self.q_models.no_models).repeat(
        #     self.batch_size)
        # # loss = torch.sum(loss)
        # loss = F.smooth_l1_loss(stsave_figate_action_values.squeeze(),
        #                         expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_models.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()
        return loss_item

    def update_target_network(self):
        self.target_models.load_state_dict(self.q_models.state_dict())

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


def initialize_model(model_params, train_hyperbolic, k, gamma, num_models, replay_buffer, batch_size, lr, loss_type,
                     target_update=False):
    target_list = []
    q_val_list = []
    if train_hyperbolic:
        return Agent(q_val_list, target_list, train_hyperbolic, k, gamma, model_params, replay_buffer, batch_size,
                     model_params.inp_dim, lr, num_models, model_params.act_space, model_params.hidden_size, loss_type,
                     target_update)
    else:
        q_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
        target_model = DQN(model_params.inp_dim, model_params.act_space, model_params.hidden_size)
        target_model.load_state_dict(q_model.state_dict())
        return Agent(q_model, target_model, train_hyperbolic, k, gamma, model_params, replay_buffer, batch_size,
                     model_params.inp_dim)


def test(agent, num_episodes=10, ep_no=-1):
    cumulative_rewards = []
    for ep in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = False
        cum_reward = 0
        i = 0
        while not done:
            i += 1
            # Select and perform an action
            action = agent.get_action(state, epsilon=0, get_among_last=False)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            state = next_state
        cumulative_rewards.append(cum_reward)
    # print("Cumulative reward on episode", ep_no, " ", sum(cumulative_rewards) / num_episodes)
    return sum(cumulative_rewards) / num_episodes


def train(num_episodes, glie_a, agent, save_fig):
    cumulative_rewards = []
    loss = None
    all_losses = []
    cumulative_rewards_test = []
    for ep in range(num_episodes):
        if ep % 100 == 0:
            print("{} episodes elapsed out of {}".format(ep, num_episodes))
        # Initialize the environment and state
        state = env.reset()
        done = False
        eps = max(0.1, glie_a / (glie_a + ep))
        # eps = min(max(eps, 0.1), 0.5)
        cum_reward = 0
        total_loss = 0
        i = 0
        while not done:
            i += 1
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            if done:
                agent.store_transition(state, action, next_state, reward, done)
            elif np.random.rand() > 0.3:
                agent.store_transition(state, action, next_state, reward, done)
            loss = agent.update_network()
            if loss is not None:
                total_loss += loss
            state = next_state
        all_losses.append(total_loss / i)
        cumulative_rewards.append(cum_reward)
        cumulative_rewards_test.append(test(agent, num_episodes=1, ep_no=ep))
        if ep == num_episodes - 1 or args.render_test:
            save_fig = None if args.render_test and ep < (num_episodes - 1) else save_fig
            plot_rewards(cumulative_rewards_test, total_losses=all_losses, save_fig=save_fig)
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network()


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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train for")
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
    parser.add_argument("--k", type=float, default=0.02, help="Learn sigma as a model parameter")
    parser.add_argument("--batch_size", type=int, default=32, help="Learn sigma as a model parameter")
    parser.add_argument("--replay_buffer", type=int, default=50000, help="Learn sigma as a model parameter")
    parser.add_argument("--num_models", type=int, default=50, help="Number of models to be used for ")
    parser.add_argument("--train_hyperbolic", default=True, action="store_true",
                        help="Train using hyperbolic discounting")
    parser.add_argument("--target_update", default=False, action="store_true",
                        help="Train using hyperbolic discounting")
    parser.add_argument("--save_fig", type=str, help="Name of the training plot")
    parser.add_argument("--loss_type", type=str, default="separate_summarized_loss")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    env = initialize_env(args.env)
    model_params, training_params = initialize_model_and_training_params(args, env)

    agent = initialize_model(model_params, args.train_hyperbolic, args.k, args.gamma, args.num_models,
                             args.replay_buffer, args.batch_size, args.lr, args.loss_type, args.target_update)
    agent.q_models.double()
    train(args.num_episodes, args.glie_a, agent, args.save_fig)
