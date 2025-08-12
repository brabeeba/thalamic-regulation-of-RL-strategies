import numpy as np
from scipy.special import logsumexp
from hmmlearn.hmm import CategoricalHMM
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch

stimuli_num = 2
context_num = 2
class_num = 2
max_episode = 500

# class MAP(nn.Module):
# 	"""docstring for MAP"""
# 	def __init__(self, H = 0.05):
# 		super(MAP, self).__init__()
# 		self.theta = nn.Parameter(0.5 * torch.ones((2, 2, 2)))
# 		self.H = H

# 	def forward(self, stimulus, action, reward):
		
# 		max_trial = stimulus.size(1)
# 		print(stimulus.size(1))
# 		temp_list = []
# 		for i in range(max_trial):
# 			print("hi")
# 			if i == 0:
# 				break
# 			temp = 0
		
# 			for j in range(max_trial):
# 				if j < i:
# 					temp += torch.log((1-reward[j]) * self.theta[0, stimulus[j], action[j]]  + reward[j]  * self.theta[0, stimulus[j], action[j]]   )
# 				else:
# 					temp += torch.log((1-reward[j]) * self.theta[1, stimulus[j], action[j]]  + reward[j] * self.theta[1, stimulus[j], action[j]])
# 			temp += i * torch.log(1-self.H) + torch.log(self.H)
# 			temp_list.append(temp)


# 		print(temp_list)
# 		temp = torch.cat(temp_list)
# 		print(temp.size())
# 		return -torch.logsumexp(temp)

# 	def __getitem__(self, idx):
# 		return self.stimulus[idx], self.action[idx], self.reward[idx]

def model_free(stimulus, action, reward):

	distance = 0
	temperature = 25
	prob = 0.5 * np.ones( (stimuli_num, class_num))
	count = np.zeros((stimuli_num, class_num))
	max_trial = stimulus.shape[0]

	for i in range(0, 50):
		distance += -temperature * prob[stimulus[i], action[i]] + logsumexp([temperature * prob[stimulus[i], :]])
		prob[stimulus[i], action[i]] += np.maximum( 0.2, 1 / (4+count[stimulus[i], action[i]])) * (reward[i] - prob[stimulus[i], action[i]])
		count[stimulus[i], action[i]] += 1

	return distance

def model_based(stimulus, action, reward):
	max_trial = stimulus.shape[1]
	H = 1 / 25
	m = CategoricalHMM(n_components = 2, init_params = "")
	m.startprob_ = np.array([1., 0.])
	m.transmat_ = np.array([[ 1-H, H], [H, 1-H]])

	mat1 = np.zeros((stimuli_num, class_num, 2))
	mat1[0, 0, 0] = 0.3
	mat1[0, 0, 1] = 0.7
	mat1[0, 1, 0] = 0.7
	mat1[0, 1, 1] = 0.3
	mat1[1, :, :] = 1- mat1[0, :, :]

	mat2 = 1-mat1

	mat1 /= 4
	mat2 /= 4

	m.emissionprob_ = np.array([mat1.flatten(),mat2.flatten()])

	data  = stimulus * 4 + action * 2 + reward * 1
	state =  m.predict_proba(data.astype(np.int32))
	
	distance = 0
	temperature = 25

	for i in range(0, 50):
		mix = state[i, 0] * mat1 +  state[i, 1] * mat2
		distance += -temperature * mix[stimulus[0,i], action[0,i], 1] + logsumexp([temperature * mix[stimulus[0, i], :, 1]])
	return distance
	







# def map_estimate(stimulus, action, reward, H = 0.05, iter = 1000):
# 	m = MAP()
# 	dataloader = DataLoader(MAPDataset(stimulus, action, reward), batch_size= 1, shuffle=False)
# 	optimizer = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=0.001)
# 	for i in range(iter):
# 		for t, (s, a , r) in enumerate(dataloader):
# 			trial_size = s.shape[0]
# 			s = torch.autograd.Variable(s).float()
# 			a = torch.autograd.Variable(a).float()
# 			r = torch.autograd.Variable(r).float()

# 			loss = m(s, a, r)
# 			optimizer.zero_grad()
# 			loss.backward()
# 			nn.utils.clip_grad_norm_(m.parameters(), 1)
# 			optimizer.step()

# 		if i % 100 == 0:
# 			print("loss {}".format(loss))
# 			print(m.theta)

# 	return m.theta

# stimulus = pd.read_csv("stimuli.csv").values[:, 1:]
# action = pd.read_csv("action.csv").values[:,1:]
# reward = pd.read_csv("reward.csv").values[:, 1:]

# print(stimulus.shape)


# model_based(stimulus[0:1], action[0:1], reward[0:1])

