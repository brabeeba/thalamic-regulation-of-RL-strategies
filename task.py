import numpy as np
import logging

logger = logging.getLogger('Bandit')

class DynamicBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "DynamicBandit"):
		super(DynamicBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		#self.prob = np.array([[[0.3, 0.4, 0.5, 0.6, 0.7]]])
		#self.prob = np.array([[[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]]])
		#self.prob = np.array([[[0.3, 0.7]]])
		self.prob = np.array([[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])

	
	def setprob(self, prob):
		self.prob = prob
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))


	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]
		
		if self.trial % self.opt["block_size"] == 0:
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		self.stimuli = np.random.randint(0, self.stimuli_num)

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")
