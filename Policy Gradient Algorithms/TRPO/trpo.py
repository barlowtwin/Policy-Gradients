import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

import gym
import numpy
import matplotlib.pyplot as plt
from collections import namedtuple


# [position of cart, velocity of cart, angle of pole, rotation rate of pole]. 
# no need of pixel space for cartpole

env = gym.make('CartPole-v1')
inp_state_size = env.observation_space.shape[0] # 4
num_actions = env.action_space.n
Rollout = namedtyuple('Rollout', ['states','actions', 'rewards', 'next_state'])


# policy networks

hidden_size = 32
actor = nn.Sequential(nn.Linear(inp_state_size, hidden_size),
					  nn.ReLU(),
					  nn.Linear(hidden_size, num_actions),
					  nn.Softmax(dim = 1))

# critic acts as a value fuction
critic = nn.Sequential(nn.Linear(inp_state_size, hidden_size),
					   nn.ReLU(),
					   nn.Linear(hidden_size, 1))


def get_action(state):
	state = torch.tensor(state).float().unsqueeze(0) # batch with single element
	dist = Categorical(actor(state)) # probability of action
	return dist.sample().item() # samples and returns a action


def update_critic(advantages):
