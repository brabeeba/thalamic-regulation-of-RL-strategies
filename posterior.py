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
	


