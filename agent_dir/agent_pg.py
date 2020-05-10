import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []

        # saved action probability
        self.action_log_probs = []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        m = Categorical(probs)
        #action = self.env.action_space.sample() # TODO: Replace this line!
        action = m.sample()
        self.action_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        #print(len(self.rewards))
        R = 0
        discount_reward = []
        for idx in range(len(self.rewards)-1,-1,-1):
            R = self.rewards[idx] + self.gamma * R
            discount_reward.append(R)
        discount_reward.reverse()
        discount_reward = torch.tensor(discount_reward)
        means = discount_reward.mean()
        stds = discount_reward.std()
        normalized_discount_reward = (discount_reward - means) / (stds + self.eps)


        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        loss = []
        # vectorize it
        for log_prob, R in zip(self.action_log_probs, normalized_discount_reward):
            loss.append(-log_prob * R)

        TotalLoss = torch.cat(loss).sum()   
        #print('TotalLoss', TotalLoss)

        self.optimizer.zero_grad()
        TotalLoss.backward(retain_graph=True)
        self.optimizer.step()

    def train(self):
        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                
                if avg_reward > 0:
                    exit(0)

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
