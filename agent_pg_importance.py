import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Bernoulli

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)

        self.action_layer = nn.Linear(hidden_dim, action_num)
        self.state_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.state_layer(x)
        action_value = self.action_layer(x)
        action_prob = F.softmax(action_value, dim=1)
        
        return state_value, action_prob

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

        # saved trajs
        self.trajs = []

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.state_values = []

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
            state_value, probs = self.model(state)
        else:
            state_value, probs = self.model(state)
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        m = Categorical(probs)
        #action = self.env.action_space.sample() # TODO: Replace this line!
        action = m.sample()
        #self.action_log_probs.append(m.log_prob(action))
        #self.state_values.append(state_value.squeeze(1))

        return m, action

    def cal_log_prob(self, state, action):

        probs = self.model(state)
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()

        return(log_prob)


    def update(self, trajs):
        t_states = []
        t_actions = []
        t_rewards = []
        t_log_probs = []
        for traj in trajs:
            t_states.append(traj[0])
            t_actions.append(traj[1])
            t_rewards.append(traj[2])
            t_log_probs.append(traj[3])
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        #print(len(self.rewards))
        t_Rs = []
        R = 0
        discount_reward = []
        for rewards in t_rewards:
            for r in rewards[::-1]:
                R = r + self.gamma * R
                discount_reward.insert(0, R)
            t_Rs.append(discount_reward)
        
        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                #state_value, action_prob = self.model(torch.from_numpy(states[t]).unsqueeze(0).to(self.device))
                m, action = self.make_action(states[t])
                log_prob = m.log_prob(action)
                print(log_prob)
                """
                #p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                """
                p_log_prob += m.log_prob(torch.tensor(actions[t]).to(self.device))
                q_log_prob += log_probs[t]
            #print(p_log_prob)
            #print(q_log_prob)
            #print(torch.exp(q_log_prob))
            
            Z_ = torch.exp(p_log_prob) / torch.exp(q_log_prob)
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
            #print(Z_)

        b = b / Z

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = 0.

            for t in range(len(Rs)):
                loss = loss - (log_probs[t] * ((Rs[t] - b).expand_as(log_probs[t]))).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)
        
        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        """
        discount_reward = torch.tensor(discount_reward).to(self.device)
        # discount reward
        normalized_discount_reward = (discount_reward - discount_reward.mean()) / (discount_reward.std() + self.eps)
        #torch.clamp_(normalized_discount_reward, -1, 1)
        #normalized_discount_reward = torch.clamp(normalized_discount_reward, -1, 1)

        state_value_tensor = torch.cat(self.state_values)
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
        """



    def train(self):
        import matplotlib.pyplot as plt
        total_reward_in_episode = 0
        window_size = 20 # size of window of moving average
        moving_reward = [] # compute moving average
        plot_list = [0] * window_size

        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                self.state_values.append(state)

                m, action = self.make_action(state)
                log_prob = m.log_prob(action)
                action = action.item()
                state, reward, done, _ = self.env.step(action)

                
                self.saved_actions.append(action)
                self.rewards.append(reward)
                self.action_log_probs.append(log_prob)

                total_reward_in_episode += reward

            self.trajs.append((self.state_values, self.saved_actions, self.rewards, self.action_log_probs))
            # update model
            self.update(self.trajs)

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
                plt.plot(plot_list)
                plt.xlabel('number of episodes playing')
                plt.ylabel('average reward of last {} episodes'.format(window_size))
                plt.title('learning curve of pg with lunar lander')
                plt.savefig('pg-learning_curve.png')