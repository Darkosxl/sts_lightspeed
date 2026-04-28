import slaythespire as s                                                                                                  
import numpy as np
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import deque

def main():

    VPG(VPG_policy_net(nstate=norm.shape[0], nhidden=20, naction=interface.getNumActions()), VPG_value_net(nstate=norm.shape[0], nhidden=20))


def step():
    pass

def reset():
    pass
def VPG_value_net(nn.module):
    def __init__(self, nstate, nhidden):
        super().__init__()
        self.hidden = nn.Linear(nstate, nhidden)
        self.value_head = nn.Linear(nhidden, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        value = self.value_head(x)
        return value.squeeze(-1)
    
def VPG_policy_net(nn.module):
    def __init__(self, nstate, nhidden, naction):
        super(VPG_policy_net, self).__init__()
        self.hidden = nn.Linear(nstate, nhidden)
        self.action_head = nn.Linear(nhidden, naction)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.action_head(x)
        return F.softmax(x, dim=1)

    def VPG_step(action):
        pass

    def VPG_render():
        pass

def VPG(policy, value_function):
    current_policy = policy
    current_value_function = value_function
    interface = s.getNNInterface()

    optimizer = torch.optim.Adam(current_policy.parameters())
    value_optimizer = torch.optim.Adam(current_value_function.parameters())
    
    gamma = 0.99
    n_episode = 1
    returns = deque(maxlen=100)
    done = False


    while True:
        rewards = []
        actions = []
        states = []
        values = []
        deltas = []

        game = s.GameContext()
        state = np.array(interface.getObservation(game), dtype=np.float32)
        state_max = np.array(interface.getObservationMax(game), dtype=np.float32)
        norm = state / state_max
        
        
        while True:

            probabilities = current_policy(torch.tensor(norm).unsqueeze(0).float())
            sampler = Categorical(probabilities)
            action = sampler.sample()
            last_value = 0.0 if done else current_value_function(torch.tensor(norm).unsqueeze(0).float()).item()
            next_state, reward, done = current_policy.VPG_step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(last_value)

            state = next_state
            if done:
                break
        values.append(0.0)

        rewards = np.array(rewards, dtype=np.float32)                                                                            
        values  = np.array(values,  dtype=np.float32)           # len T+1                                                        
        states  = np.array(states,  dtype=np.float32)                                                                            
        actions = np.array(actions, dtype=np.int64)
        
        
        R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))]).float()
        deltas = rewards + gamma * values[1:] - values[:-1]
        lam = 0.95
        A = np.zeros_like(deltas)
        accumulator = 0.0
        for t in reversed((range(len(deltas)))):
            accumulator = deltas[t] + gamma * lam * accumulator
            A[t] = accumulator
        A = (A - A.mean()) / (A.std() + 1e-8)
        A = torch.tensor(A, dtype=torch.float32)

        states = torch.tensor(states).float()
        actions = torch.tensor(actions)

        probabilities = current_policy(states)
        sampler = Categorical(probabilities)
        log_probabilities = sampler.log_prob(actions)
        loss = -(log_probabilities * A).mean()

        for i in range(80):
            value_optimizer.zero_grad()
            value_loss = ((current_value_function(states) - R) ** 2).mean()
            value_loss.backward()
            value_optimizer.step()
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        returns.append(np.sum(rewards))
        n_episode += 1

    env.close()