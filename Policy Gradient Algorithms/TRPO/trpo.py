import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


# [position of cart, velocity of cart, angle of pole, rotation rate of pole]. 
# no need of pixel space for cartpole

env = gym.make('CartPole-v1')
inp_state_size = env.observation_space.shape[0] # 4
num_actions = env.action_space.n
Rollout = namedtuple('Rollout', ['states','actions', 'rewards', 'next_state'])
max_d_kl = 0.01 # delta upper bound for KL divergence


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
critic_optimizer = Adam(critic.parameters(), lr = 0.005)


def get_action(state):
	state = torch.tensor(state).float().unsqueeze(0) # batch with single element
	dist = Categorical(actor(state)) # probability of action
	return dist.sample().item() # samples and returns a action


def update_critic(advantages):

	# mse loss for advantage
	loss = 0.5 * (advantages ** 2).mean()
	critic_optimizer.zero_grad()
	loss.backward()
	critic_optimizer.step()


def estimate_advantages(states, last_state, rewards):
	values = critic(states)
	last_value = critic(last_state.unsqueeze(0))
	next_values = torch.zeros_like(rewards)
	for i in reversed(range(rewards.shape[0])):
		last_value = next_values[i] = rewards[i] + 0.99 * last_value
	advantages = next_values - values
	return advantages


def surrogate_loss(new_probs, old_probs, advantages):
	return (new_probs / old_probs * advantages).mean()


def kl_div(p, q):
	p = p.detach()
	return (p * (p.log() - q.log())).sum(-1).mean()


##############################################################################

def train(epochs = 100, num_rollouts = 10, render_frequency = None):

	mean_total_rewards = []
	global_rollout = 0

	for epoch in range(epochs):
		rollouts = []
		rollout_total_rewards = []


		for t in range(num_rollouts):
			state = env.reset()
			done = False
			samples = [] # every tuple till done is appended to sample

			while not done :
				if render_frequency is not None and global_rollout % render_frequency == 0:
					env.render()

				with torch.no_grad():
					action = get_action(state)

				next_state, reward, done, _  = env.step(action)
				samples.append((state, action, reward, next_state))
				state = next_state

			states, actions, rewards, next_states = zip(*samples)

			states = torch.stack([torch.from_numpy(state) for state in states], dim = 0).float()
			next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim = 0).float()
			actions = torch.tensor(actions).unsqueeze(1)
			rewards = torch.tensor(rewards).unsqueeze(1)

			rollouts.append(Rollout(states, actions, rewards, next_states))
			rollout_total_rewards.append(rewards.sum().item())
			global_rollout += 1

		update_agent(rollouts)
		mtr = np.mean(rollout_total_rewards)
		print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')
		mean_total_rewards.append(mtr)

	# plt.plot(mean_total_rewards)
	# plt.show()

###############################################################################

def flat_grad(y, x, retain_graph = False, create_graph = False):

	if create_graph:
		retain_graph = True

	g = torch.autograd.grad(y, x, retain_graph = retain_graph, create_graph = create_graph)
	g = torch.cat([t.view(-1) for t in g])
	return g



def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel



def conjugate_gradient(A, b, delta = 0., max_iterations = 10):

	# A is a function that multiples d_kl with a matrix and returns the
	# output of the gradient wrt actor parameters

	# b is tensor of gradients of surorgate loss wrt actor parameters
	# b has size equal to number of parameters

	x = torch.zeros_like(b) # num_parameters
	r = b.clone() # num_parameters
	p = b.clone() # num_parameters


	i = 0
	while i < max_iterations:

		AVP = A(p) # gradient of surrogate loss wrt actor params is multiplied by p
		# num_parameters

		dot_old = r @ r
		alpha = dot_old / (p @ AVP)
		x_new = x + alpha * p

		if (x - x_new).norm() <= delta:
			return x_new # num_parameters

		i += 1
		r = r - alpha * AVP
		beta = (r @ r) / dot_old
		p = r + beta * p
		x = x_new


	return x # num_parameters



def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()

    update_critic(advantages)

    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]

    # Now we have all the data we need for the algorithm

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1



train(epochs = 50, num_rollouts = 10, render_frequency = 50)
