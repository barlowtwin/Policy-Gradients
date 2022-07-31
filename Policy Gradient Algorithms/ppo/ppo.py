import numpy as np 
import torch
from torch.distributions.categorical import Categorical 

class Memory:

	def __init__(self, batch_size):

		self.states = []
		self.actions = []
		self.rewards = []
		self.done = []
		self.probs = []
		self.vals = []
		self.batch_size = batch_size

	def generate_batches(self):

		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size) # slices equivalent to batch_size
		indices = np.arange(n_states, dtype = np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.states), np.array(self.actions), np.array(self.rewards),\
				np.array(self.done), np.array(self.probs), np.array(self.vals)


