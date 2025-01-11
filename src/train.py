from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient
import numpy as np
import torch
import pickle
import torch.nn as nn
from joblib import dump, load
import random
import gymnasium as gym
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from gymnasium.wrappers import TimeLimit
import pickle
from copy import deepcopy
    

env = TimeLimit(env=HIVPatient(domain_randomization=False), #False
                max_episode_steps=200)  

# env = TimeLimit(env=FastHIVPatient(domain_randomization=False), #False
#                 max_episode_steps=200)  


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device
    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.buffer)

######################
### Random forest ####
######################

class RandomForestFQI():
    def __init__(self, gamma=0.7, horizon=200, iterations=1000, replay_buffer_capacity=10000, batch_size=64):
        # self.env = TimeLimit(
        #     env=FastHIVPatient(domain_randomization=False),
        #     max_episode_steps=200
        # )
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=True), # False
            max_episode_steps=200
        )
        self.states = gym.spaces.Discrete(4)
        self.actions = [np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]]
        self.nb_actions = len(self.actions)
        self.gamma = gamma
        self.rf_model = None
        self.horizon = horizon
        self.iterations = iterations
        self.batch_size = batch_size
        
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, device='cpu')

    def collect_samples(self, disable_tqdm=False, print_done_states=False):
        s, _ = self.env.reset()
        for _ in tqdm(range(self.horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            self.replay_buffer.append(s, a, r, s2, done)
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2

    def train(self, disable_tqdm=False):
        self.collect_samples()
        Qfunctions = []

        for iter in tqdm(range(self.iterations), disable=disable_tqdm):
            # Sample mini-batch from the replay buffer
            if len(self.replay_buffer) < self.batch_size:
                print("Not enough samples in the replay buffer.")
                break

            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch

            # Prepare the feature matrix
            SA = np.concatenate((states, actions), axis=1)

            if iter == 0:
                value = rewards.copy()
            else:
                Q2 = np.zeros((self.batch_size, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2 * np.ones((self.batch_size, 1))
                    S2A2 = np.concatenate((next_states, A2), axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = rewards + self.gamma * (1 - dones) * max_Q2
            
            # Train the Q-function using the RandomForestRegressor
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
            
            # Average reward for monitoring
            avg_reward = np.mean(value)
            print(f"Iteration {iter + 1}/{self.iterations}, Average Reward: {avg_reward}")
        
        # Save the trained model
        self.rf_model = Qfunctions[-1]
        with open("src/random_forest_model.pkl", "wb") as file:
            pickle.dump(Qfunctions[-1], file)

# agent = RandomForestFQI()
# agent.train(env)

    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

######################
###      DQN      ####
######################

class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def MC_eval(self, env, nb_trials):   
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   
        MC_avg_discounted_reward = []   
        V_init_state = []   
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        max_reward = -float('inf') 
        
        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if self.monitoring_nb_trials > 0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   
                    MC_avg_total_reward.append(MC_tr)   
                    MC_avg_discounted_reward.append(MC_dr)  
                    V_init_state.append(V0)   
                    episode_return.append(episode_cum_reward)   
                    print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                        ", MC tot ", '{:6.2f}'.format(MC_tr),
                        ", MC disc ", '{:6.2f}'.format(MC_dr),
                        ", V0 ", '{:6.2f}'.format(V0),
                        sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                        sep='')

                # Check and save the best model
                if episode_cum_reward > max_reward: 
                    max_reward = episode_cum_reward
                    print(f"New max reward achieved: {max_reward:.2e}. Saving model...")
                    torch.save(self.model.state_dict(), "src/best_model_9.pth")
                
                # Reset environment for next episode
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state


######################
###### Network #######
######################

state_dim = 6  
nb_neurons = 256 # 256, 128, 64
n_action = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DQN = nn.Sequential(
    nn.Linear(state_dim, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, n_action)
).to(device)


### BEST CONFIG

config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.90,
                'buffer_size': 10000, # 40000, 30000, 20000
                'epsilon_min': 0.05, # 0.1, 0.08
                'epsilon_max': 1.0,
                'epsilon_decay_period': 40000, #30000, 20000
                'epsilon_delay_decay': 250, # 200, 300, 400
                'batch_size': 1000, # 2000, 500
                'gradient_steps': 2,
                'update_target_strategy': 'ema', #'replace',
                'update_target_freq': 75, # 50
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss(),
                'monitoring_nb_trials': 0,
            }


######################
### Project agent ####
######################


class ProjectAgent:
    def __init__(self, name='DQN', config=config):
        # self.env = TimeLimit(
        #     env=FastHIVPatient(domain_randomization=False),
        #     max_episode_steps=200
        # )
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=False), #False
            max_episode_steps=200
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.original_env = self.env.env
        
        if config is None:
            config = {}
        
        if 'nb_actions' not in config:
            raise ValueError("'nb_actions' key is missing in the configuration dictionary")
        
        self.nb_actions = config['nb_actions']  
        
        self.dqn_agent = dqn_agent(config=config, model=self._create_dqn_model())

    def _create_dqn_model(self):
        state_dim = 6  
        nb_neurons = 256  # 512, 256, 128, 64
        n_action = 4 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        ).to(device)
        return model
    
    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        
        self.load()
        
        if self.name == 'RF_FQI':
            Qsa = []
            for a in range(self.nb_actions):
                sa = np.append(observation, a).reshape(1, -1)
                Qsa.append(self.Qfunction.predict(sa))  
            return np.argmax(Qsa)
        
        elif self.name == 'DQN':
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            Q_values = self.dqn_agent.model(state_tensor)  
            return torch.argmax(Q_values, dim=1).item() 
        else:
            raise ValueError("Unknown model")
        
    def load(self):
        if self.name == 'RF_FQI':
            self.load_rf_fqi() 
        elif self.name == 'DQN':
            self.load_dqn()
        else:
            raise ValueError("Unknown model type") 

    def load_rf_fqi(self):
        model_path = "src/random_forest_model.pkl"
        with open(model_path, 'rb') as file:
            self.Qfunction = pickle.load(file)

    def load_dqn(self):
        model_path = "src/best_model_9.pth"

        state_dict = torch.load(model_path, map_location=self.device)
        self.dqn_agent.model.load_state_dict(state_dict)
        self.dqn_agent.model.eval() 


# # Train agent
# agent = dqn_agent(config, DQN)
# ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 400)


######################
###### TRAINING ######
######################


if __name__ == "__main__":
    agent = dqn_agent(config, DQN)
    ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 400)