import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Bernoulli

from agent_dir.agent import Agent
from environment import Environment
from torch.utils.tensorboard import SummaryWriter

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc(x))
        action_value = self.action_layer(x)
        action_prob = F.softmax(action_value, dim=1)
        
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'

        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=128).to(self.device)
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
        self.action_log_probs = []
        self.state_values = []

    def make_action(self, state, test=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if test == True:
            self.model.eval()
            probs = self.model(state)
        else:
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
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discount_reward.insert(0, R)
        
        discount_reward = torch.tensor(discount_reward).to(self.device)
        # discount reward
        normalized_discount_reward = (discount_reward - discount_reward.mean()) / (discount_reward.std() + self.eps)
        #torch.clamp_(normalized_discount_reward, -1, 1)
        #normalized_discount_reward = torch.clamp(normalized_discount_reward, -1, 1)

        #print(len(state_value_tensor), len(normalized_discount_reward))

        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        
        # vectorize it
        self.action_log_probs = torch.cat(self.action_log_probs)
        PG_Loss_vector = - self.action_log_probs * discount_reward
        PG_Loss_vector = PG_Loss_vector.sum()

        self.optimizer.zero_grad()
        PG_Loss_vector.backward()
        self.optimizer.step()



    def train(self):
        import matplotlib.pyplot as plt
        total_reward_in_episode = 0
        window_size = 20 # size of window of moving average
        moving_reward = [] # compute moving average
        plot_list = [0] * window_size

        avg_reward = None

        #tb = SummaryWriter(log_dir='./runs')

        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)

                #tb.add_graph(self.model, torch.from_numpy(state).float().unsqueeze(0).to(self.device))

                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

                total_reward_in_episode += reward

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                

            if avg_reward > 200: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
            
            if len(moving_reward) >= window_size:
                moving_reward.pop(0)
            moving_reward.append(total_reward_in_episode)
            total_reward_in_episode = 0

            # plot moving average reward
            if len(moving_reward) >= window_size:
                plot_list.append(sum(moving_reward)/len(moving_reward))
                yield plot_list
                """
                plt.plot(plot_list)
                plt.xlabel('number of episodes playing')
                plt.ylabel('average reward of last {} episodes'.format(window_size))
                plt.title('learning curve of pg with lunar lander')
                plt.savefig('pg-learning_curve.png')
                """
        #tb.close()