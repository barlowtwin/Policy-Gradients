import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

import numpy as np
import gym
from gym.spaces import Discrete, Box

def Network(inp_dim, hidden_dim, act_dim):

	return nn.Sequential(nn.Linear(inp_dim, hidden_dim),
						nn.ReLU(),
						nn.Linear(hidden_dim, act_dim),
						nn.Softmax(dim = 0))



def train(env_name = 'CartPole-v0', hidden_dim = 32, lr = 1e-3, epochs = 50, batch_size = 5000, render = False):

	env = gym.make(env_name)
	assert isinstance(env.observation_space, Box), "this environment only works with continuous spaces"
	assert isinstance(env.action_space, Discrete), "this environment only works with discrete action spaces"

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n
	policy = Network(obs_dim, hidden_dim, act_dim)
	optimizer = Adam(policy.parameters(), lr = lr)

	def get_action(obs):
		dist = Categorical(policy(obs))
		return dist.sample().item()

	def compute_loss(obs, actions, weights): # weights is R_tau
		dist = Categorical(policy(obs))
		logp = dist.log_prob(actions)
		return -(logp * weights).mean() 

	def train_one_epoch():

		batch_obs = [] # obs
		batch_acts = [] # actions
		batch_weights = [] # R_tau
		batch_rets = [] # episode returns
		batch_lens = [] # episode lengths

		obs = env.reset()
		done = False
		ep_rews = []


		while True :

			if render :
				env.render()


			batch_obs.append(obs)

			# act in the env
			act = get_action(torch.tensor(obs, dtype = torch.float32))
			obs, rew,  done, _ = env.step(act)

			# save action and reward
			batch_acts.append(act)
			ep_rews.append(rew)

			if done :

				# if episode is over
				ep_ret, ep_len = sum(ep_rews), len(ep_rews)
				batch_rets.append(ep_ret)
				batch_lens.append(ep_len)

				# weight for log_probs
				batch_weights += [ep_ret] * ep_len

				# reset episode specific variables
				obs = env.reset()
				done = False
				ep_rews = []


				# stop experience loop if we have more than batch_size
				if len(batch_obs) > batch_size :
					break

		optimizer.zero_grad()
		batch_loss = compute_loss( obs = torch.tensor(batch_obs, dtype = torch.float32),
									actions = torch.tensor(batch_acts, dtype = torch.int32),
									weights = torch.tensor(batch_weights, dtype = torch.float32))
		batch_loss.backward()
		optimizer.step()
		return batch_loss, batch_rets, batch_lens

	# training loop
	for i in range(epochs):
		batch_loss, batch_rets, batch_lens = train_one_epoch()
		print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)




