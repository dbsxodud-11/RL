import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

#Soft Update
def update_model(source, target, tau) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

#ReplayMemory
class ReplayMemory :

    def __init__(self, capacity) :
        self.memory = deque(maxlen = capacity)
    
    def __len__(self) :
        return len(self.memory)

    def push(self, transition) :
        self.memory.append(transition)

    def sample(self, batch_size) :

        transitions = random.sample(self.memory, batch_size)

        state, action, reward, next_state, done = zip(*transitions)

        state = torch.cat(state)
        action = torch.tensor(action, dtype=torch.float32).reshape(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1, 1)
        next_state = torch.cat(next_state)
        done = torch.tensor(done, dtype=torch.long).reshape(-1, 1)

        return state, action, reward, next_state, done

#Actor(state -> action)
class Actor(nn.Module) :

    def __init__(self, state_dim, action_dim, hidden_dim) :
        super(Actor, self).__init__()

        input_dims = [state_dim] + hidden_dim
        output_dims = hidden_dim + [action_dim]
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for in_dim, out_dim in zip(input_dims, output_dims) :
            self.layers.append(nn.Linear(in_dim, out_dim))
        
        for _ in range(len(hidden_dim)) :
            self.activations.append(nn.LeakyReLU())
        self.activations.append(nn.Tanh())

    def forward(self, x) :

        for layer, activation in zip(self.layers, self.activations) :
            x = layer(x)
            x = activation(x)
        return x

#Critic(state, action -> Q value)
class Critic(nn.Module) :

    def __init__(self, state_dim, action_dim, hidden_dim) :
        super(Critic, self).__init__()

        input_dims = [state_dim + action_dim] + hidden_dim
        output_dims = hidden_dim + [action_dim]
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for in_dim, out_dim in zip(input_dims, output_dims) :
            self.layers.append(nn.Linear(in_dim, out_dim))

        for _ in range(len(hidden_dim)) :
            self.activations.append(nn.LeakyReLU())
        self.activations.append(nn.Identity())

    def forward(self, state, action) :
        
        x = torch.cat([state, action], dim=1)
        for layer, activation in zip(self.layers, self.activations) :
            x = layer(x)
            x = activation(x)
        return x

#OU noise
class OUNoise:
    """
    Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=1000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

        self.epsilon = 1.0
        self.epsilon_decay = 0.0003
        self.epsilon_min = 0.05

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state() * self.epsilon

        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)

        return np.clip(action + ou_state, -1.0, 1.0)

#DDPG Agent
class DDPGAgent(nn.Module) :

    def __init__(self, state_dim, action_dim, action_min, action_max, gamma=0.99) :
        super(DDPGAgent, self).__init__()

        self.replay_memory = ReplayMemory(5000)
        self.ou_noise = OUNoise(action_space=1)
        self.actor = Actor(state_dim, action_dim, hidden_dim = [128 for _ in range(4)])
        self.critic = Critic(state_dim, action_dim, hidden_dim = [128 for _ in range(3)])
        self.target_actor = Actor(state_dim, action_dim, hidden_dim = [128 for _ in range(4)])
        self.target_critic = Critic(state_dim, action_dim, hidden_dim = [128 for _ in range(3)])
        update_model(self.actor, self.target_actor, tau=1.0)
        update_model(self.critic, self.target_critic, tau=1.0)

        self.action_min = action_min
        self.action_max = action_max
        self.gamma = gamma
        self.batch_size = 64
        self.tau = 0.005

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def get_action(self, state, t=0) :

        action_before_norm = self.actor(state).detach().numpy()
        action_after_norm = self.ou_noise.get_action(action_before_norm, t)
        action_after_norm = (action_after_norm + 1)/2 * (self.action_max - self.action_min) + self.action_min

        return action_before_norm, action_after_norm

    def push(self, transition) :
        self.replay_memory.push(transition)

    def train_start(self) :
        return len(self.replay_memory) >= self.batch_size

    def train(self) :

        state, action, reward, next_state, done = self.replay_memory.sample(self.batch_size)

        #1. Critic Update
        current_q_values = self.critic(state, action)
        next_q_values = self.target_critic(next_state, self.target_actor(next_state))
        target = reward + self.gamma * next_q_values.detach() * (1 - done)

        value_loss = self.criterion(target, current_q_values)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        #2. Actor Update
        policy_loss = -self.critic(state, self.actor(state))
        policy_loss = policy_loss.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        #3. Target Update
        update_model(self.actor, self.target_actor, tau = self.tau)
        update_model(self.critic, self.target_critic, tau = self.tau)

        return value_loss.item(), policy_loss.item()

def main_DDPG() :

    env = gym.make("Pendulum-v0")
    max_episode = 200

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_min = env.action_space.low
    action_max = env.action_space.high

    agent = DDPGAgent(state_dim, action_dim, action_min, action_max)

    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    for episode in range(max_episode) :

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

        reward_epi = []
        actor_loss_epi = []
        critic_loss_epi = []
        done = False
        step = 0

        while not done :    

            action, action_norm = agent.get_action(state, step)
            
            next_state, reward, done, _ = env.step(action_norm)
            next_state = torch.tensor(next_state, dtype=torch.float32).reshape(1, -1)

            reward_epi.append(reward)
            transition = [state, action, reward, next_state, done]
            agent.push(transition)
            state = next_state

            if agent.train_start() :
                critic_loss, actor_loss = agent.train()
                critic_loss_epi.append(critic_loss)
                actor_loss_epi.append(actor_loss)
            
        env.close()
        step += 1

        reward_list.append(sum(reward_epi))
        if agent.train_start() :
            critic_loss_list.append(sum(critic_loss_epi) / len(critic_loss_epi))
            actor_loss_list.append(sum(actor_loss_epi) / len(actor_loss_epi))

        print(episode+1, reward_list[-1])
    
    return actor_loss_list, critic_loss_list, reward_list

if __name__ == "__main__" :

    actor_loss, critic_loss, reward = main_DDPG()

    plt.title("Actor(Policy) Loss")
    plt.xlabel("number of episode")
    plt.plot(actor_loss)
    plt.show()
    plt.close("all")

    plt.title("Critic(Value) Loss")
    plt.xlabel("number of episode")
    plt.plot(critic_loss)
    plt.show()
    plt.close("all")

    plt.title("Reward")
    plt.xlabel("number of episode")
    plt.plot(reward)
    plt.show()
    plt.close("all")
