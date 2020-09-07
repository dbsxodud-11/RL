import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

#Update Target Networks
def update_model(source, target, tau) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

#Replay Memory
class ReplayMemory:

    def __init__(self, capacity) :
        self.memory = deque(maxlen = capacity)

    def __len__(self) :
        return len(self.memory)

    def push(self, transition) :
        self.memory.append(transition)

    def sample(self, batch_size) :
        transition = random.sample(self.memory, batch_size)

        state, action, reward, next_state, done = zip(*transition)

        state = torch.cat(state)
        action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1, 1)
        next_state = torch.cat(next_state)
        done = torch.tensor(done, dtype=torch.int32).reshape(-1, 1)

        return state, action, reward, next_state, done
        
#Neural Network        
class MLP(nn.Module) : 

    def __init__(self, input_dim, output_dim, hidden) :
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]) :
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))

    def forward(self, x) :
        for layer in self.layers :
            x = layer(x)
        return x

#DQN Agent
class DQNAgent(nn.Module) :

    def __init__(self, replay_memory, main_network, target_network, batch_size) :
        super(DQNAgent, self).__init__()

        self.replay_memory = replay_memory
        self.main_network = main_network
        self.target_network = target_network
        update_model(main_network, target_network, tau = 1.0)

        self.batch_size = batch_size
        self.gamma = 0.99

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr = 0.001)
    
    def forward(self, x) :

        x = self.main_network(x)
        return x

    def push(self, transition) :

        self.replay_memory.push(transition)

    def train_start(self) :

        return len(self.replay_memory) >= self.batch_size

    def train(self) :

        state, action, reward, next_state, done = self.replay_memory.sample(self.batch_size)
        current_q_values = self.main_network(state).gather(1, action)
        #print(current_q_values)
        next_q_values = torch.max(self.target_network(next_state), 1)[0].reshape(-1, 1).detach()
        target = reward + self.gamma * next_q_values * (1 - done)
        #print(target)
        mse_loss = self.criterion(target, current_q_values)

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()
        
        return mse_loss.item()

    def update_target(self) :

        update_model(self.main_network, self.target_network, tau=0.05)

def main_DQN():

    max_episode = 200
    env = gym.make("CartPole-v0")

    input_dim = 4
    output_dim = 2

    replay_memory = ReplayMemory(5000)
    main_network = MLP(input_dim, output_dim, hidden = [64 for _ in range(3)])
    target_network = MLP(input_dim, output_dim, hidden = [64 for _ in range(3)])

    epsilon = 0.9
    epsilon_decay = 0.05
    epsilon_min = 0.05
    batch_size = 64

    agent = DQNAgent(replay_memory, main_network, target_network, batch_size)

    loss_list = []
    reward_list = []
    steps = 1
    target_update = 5

    for episode in range(max_episode) :

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

        done = False
        reward_epi = []
        loss_epi = []
        while not done :

            if random.random() < epsilon :
                action = env.action_space.sample()
            else :
                action = torch.argmax(agent(state)).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).reshape(1, -1)

            reward_epi.append(reward)
            transition = [state, action, reward, next_state, done]
            agent.push(transition)
            state = next_state

            if(agent.train_start()) :
                loss = agent.train()
                loss_epi.append(loss)

            if steps % target_update == 0 :
                agent.update_target()

            steps += 1

        env.close()

        epsilon -= epsilon_decay
        if epsilon <= epsilon_min :
            epsilon = epsilon_min
        
        reward_list.append(sum(reward_epi))
        if agent.train_start() :
            loss_list.append(sum(loss_epi)/len(loss_epi))
        
        print(episode+1, reward_list[-1])

    return loss_list, reward_list

if __name__ == "__main__" :

    loss_list, reward_list = main_DQN()

    plt.title("Loss of action-value function")
    plt.plot(loss_list)
    plt.show()
    plt.close("all")

    plt.title("Reward")
    plt.xlabel("number of episodes")
    plt.plot(reward_list)
    plt.show()
    plt.close("all")
