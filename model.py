import numpy as np
import logging
import scipy.special as special
from util import *
from scipy.special import expit
from scipy.stats import norm

logger = logging.getLogger('Bandit')



class TwoTimeScaleNeuralAgent(object):
	"""docstring for TwoTimeScaleAgent"""
	def __init__(self, opt, name = "Thalamocortical Model"):
		super(TwoTimeScaleNeuralAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.gamma1 = opt["gamma1"]
		self.tau = opt["tau"]
		self.s = 1.0 / (2 * self.tau)
	
		self.temperature = opt["temperature"]
		self.lr = opt["lr"]
		self.a = opt["a"]
		self.name = name

		self.learning = opt["learning"]
		self.nonlinear = opt["nonlinear"]
		
		self.dt = opt["dt"]
		self.a1 = opt["a1"]
		self.b1 = opt["b1"]
		self.a2 = opt["a2"]
		self.b2 = opt["b2"]

		self.tau = opt["tau"]
		self.eta = opt["eta"]
		self.tau1 = opt["tau1"]
		self.threshold = opt["threshold"]
		self.d_interval = opt["d_interval"]
		self.K = opt["K"]

		self.inhibit = opt["inhibit"]
		self.d2 = opt["d2"]
		self.rescue = opt["rescue"]
	

		self.scan_interval = opt["scan_interval"]
		self.scan_num = self.d_interval // self.scan_interval




		self.stimuli = 0

		self.thalamus = np.zeros(self.context_num)
		self.thalamus[0] = 2 * self.a * self.tau + 2
		self.thalamus[1] = 2

		# self.thalamus[0] = 2 * self.a+ 2
		# self.thalamus[1] = 2


		if self.d2:
			self.thalamus[0] = 8

		self.vip = np.zeros(self.context_num)
		self.pv = np.zeros(self.context_num)

		self.ct = 0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))

		self.tt = -  self.s * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 


		self.quantile_num = opt["quantile_num"]
		
		self.conf = 0

		self.prob =   0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num))

		self.pfc_neurons = np.zeros((self.stimuli_num, self.class_num, 2))
	

		self.stimuli_neurons = np.zeros(self.stimuli_num)


		self.value_neurons = np.zeros((self.context_num, self.class_num))


		self.decision_neurons = np.zeros(self.class_num)
	
		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
		for i in range(self.class_num):
			self.decision_w[i, i] = self.a2

		



		self.count = np.zeros( (opt["context_num"], opt["stimuli_num"], opt["class_num"]))
		self.count1 = np.zeros((opt["context_num"], opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0



		self.R = 0
		self.r = 0

		self.time = 0
		self.confidence = 0

		self.past_choice = [0]
		self.lr = 0

		self.pfc_scan = np.zeros((self.scan_num , self.stimuli_num, self.class_num, 2))
		self.pfc_md_scan = np.zeros((self.scan_num, self.context_num, self.stimuli_num, self.class_num, 2))
		self.md_scan = np.zeros((self.scan_num, self.context_num))
		self.alm_scan = np.zeros((self.scan_num, self.stimuli_num))
		self.alm_bg_scan = np.zeros((self.scan_num, self.context_num, self.stimuli_num, self.class_num))
		self.bg_scan = np.zeros((self.scan_num, self.context_num, self.class_num))
		self.m1_scan = np.zeros((self.scan_num, self.class_num))
		self.pv_scan = np.zeros((self.scan_num, self.context_num))
		self.vip_scan = np.zeros((self.scan_num, self.context_num))


	def f_scalar(self, x):
		if x >= self.a:
			return 2 * self.a -2
		elif x <= -self.a:
			return -2
		else:
			return x + self.a - 2

	def f(self, x):
		return np.vectorize(self.f_scalar)(x)
		

	def g(self, x):
		return relu(2.7 + np.log(x))
		#return relu(3 + np.log(x))

	def h(self, x):
		return np.minimum(1, relu(x))

	def sig(self, x):
		return 1 / (1 + np.exp(-10 * x + 5))

	#1 / (1 + np.exp(-10 * x + 5)) 

	def md_f(self, x):
		#return relu(2 /  (1 + np.exp(4*self.a - 4*(x-2)))- 1)
		return relu(2 /  (1 + np.exp(4*self.a*self.tau - 4*((x+2)-2 *self.a*self.tau )))- 1)

	def in_f(self, x):
		return relu(2 /  (1 + np.exp( - 2*(x)))-1)



	def ev(self):
		return self.prob

	def get_ev(self):
		return self.prob[self.context, :, :]

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def histogram(self):
		return {"PFC": self.pfc_scan.copy(), "PFC/MD": self.pfc_md_scan.copy(), "MD": self.md_scan.copy(), "ALM": self.alm_scan.copy(), "ALM/BG": self.alm_bg_scan.copy(), "BG": self.bg_scan.copy(), "M1": self.m1_scan.copy(), "PV": self.pv_scan.copy(), "VIP": self.vip_scan.copy()}


	def scalars(self):
		hist = {}
		ev = self.ev()
		for c in range(self.context_num):
			for s in range(self.stimuli_num):
				for a in range(self.class_num):
					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
					for r in range(2):
						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.ct[c, s, a, r]
			hist["context-{}".format(c)] = self.thalamus[c]

			
		
		hist["fast-reward"] = self.r
		hist["confidence"] = self.confidence 
		hist["context-difference"] = self.thalamus[0] - self.thalamus[1]
		hist["choice_prob"] = self.get_choice_prob()
		hist["smooth_confidence"] = self.conf
		hist["learning_rate"] = self.lr
		hist["md-nonlinearity"] = self.md_f(self.thalamus[0])
		hist["in-nonlinearity"] = self.in_f(self.vip[0]-self.pv[0])
		hist["action"] = self.action
		hist["context"] = self.context

		return hist

	def forward(self, x):
		self.time += 1
		

		

		self.stimuli_neurons = np.zeros(self.stimuli_num)
		self.stimuli_neurons[x] = 1
		self.value_neurons = np.zeros((self.context_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		self.pfc_neurons = np.zeros((self.stimuli_num, self.class_num, 2))
		pfc_input = np.zeros((self.stimuli_num, self.class_num, 2))
		pfc_input[self.stimuli, self.action, self.r] = 1
		self.pfc_neurons = pfc_input


		action_bool = False


		for i in range(self.d_interval):

			if i % self.scan_interval == 0:
				idx = i // self.scan_interval
				self.pfc_scan[idx] = self.pfc_neurons.copy()
				self.pfc_md_scan[idx] = self.ct.copy()
				self.md_scan[idx] = self.thalamus.copy()
				self.alm_scan[idx] = self.stimuli_neurons.copy() 
				self.alm_bg_scan[idx] = self.prob.copy()
				self.bg_scan[idx] = self.value_neurons.copy()
				self.m1_scan[idx] = self.decision_neurons.copy()
				self.pv_scan[idx] = self.pv.copy()
				self.vip_scan[idx] = self.vip.copy()


			if self.thalamus[0] - self.thalamus[1] >= 0:
				self.context = 0
			else:
				self.context = 1


			self.vip += self.dt * self.tau1 * (-self.vip + self.thalamus)
			self.pv += self.dt * self.tau1 * (-self.pv + np.flip(self.thalamus))

			self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
			self.conf += self.dt * self.tau1 * (-self.conf + self.confidence)





			if self.nonlinear:
				
				self.thalamus += self.dt  *  (-1.0 / self.tau * self.thalamus + self.f(self.tt.dot(self.thalamus)) +  self.g(self.ct.reshape(2, 8).dot(self.pfc_neurons.reshape(8))) )
			else:
				self.thalamus += self.dt  * (-1.0 / self.tau * self.thalamus + relu(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, self.r]))

			

			if self.inhibit:
				self.thalamus = np.zeros(self.context_num)
			
			if self.learning:

				#self.stimuli_neurons += self.dt * self.tau1 * (-self.stimuli_neurons + np.eye(self.stimuli_num)[x])
				for k in range(self.context_num):
					self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  self.h(self.prob[k, :, :].T.dot(self.stimuli_neurons) * self.in_f(self.vip[k] - self.pv[k])))
				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * self.value_neurons.T.dot(np.ones(self.class_num)))
			
			else:
				for k in range(self.context_num):
				
					self.sample_neurons[x][k] +=  0.03 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
					if k == self.context:
						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T ))
					else:
						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
				
				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
			


			if np.max(self.decision_neurons) > self.threshold:
				action = np.argmax(self.decision_neurons)
				action_bool = True
		

		# print("prob context 0: {}".format(self.ct[0, self.stimuli, self.action, self.r]),  self.stimuli, self.action, self.r)
		# print("prob context 1: {}".format(self.ct[1, self.stimuli, self.action, self.r]),  self.stimuli, self.action, self.r)




		if not action_bool:
			
			#action = np.random.randint(self.class_num)
			#action = np.argmax(self.decision_neurons)
			action = np.random.choice(self.class_num, p= special.softmax(self.temperature * self.decision_neurons))
			# print(self.ev()[0, x, :])
			# print(self.ev()[1, x, :])
			# print(self.decision_neurons, self.stimuli, action)
			# print("Random decision is made {}".format(self.name))
			# print(self.decision_neurons, self.threshold, self.context)
			# print(np.sort(self.value_neurons[self.context, :, 0])[-4:])
			

		if self.learning:
			#self.count[self.context, x, action] += self.conf
			self.count[:, x, action] +=  self.in_f(self.vip-self.pv)
		else:
			self.count[self.context, x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action


		if self.time % 200 == 0:
			print(self.time)

		self.stimuli = x



		return self.action

	def update(self, r):
		self.r = r

		# self.thalamus += -1.0 / self.tau * self.thalamus + self.f(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, r])
		if self.thalamus[0] - self.thalamus[1] >= 0:
			self.context = 0
		else:
			self.context = 1
		self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
	

		if self.learning:
			#self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) * self.conf / (3+self.count[self.context, self.stimuli, self.action])
			# self.prob[:, self.stimuli, self.action, :-1] += (r - self.prob[:, self.stimuli, self.action, :-1]) * np.expand_dims(self.in_f(self.vip-self.pv), 1) / (7+np.expand_dims(self.count[:, self.stimuli, self.action], 1))
			self.prob[:, self.stimuli, self.action] += (r - self.prob[:, self.stimuli, self.action]) * self.in_f(self.vip-self.pv) * np.maximum( 0.2, 1 / (4+self.count[:, self.stimuli, self.action]))
			
			#self.count1[self.context, self.stimuli, self.action] += self.confidence
			self.count1[:, self.stimuli, self.action] += self.md_f(self.thalamus)

			inputs = np.zeros(2)
			inputs[r] = 1
			#self.ct[self.context, self.stimuli, self.action, r] +=   self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
			#self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])

			self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) * np.maximum( 0.1, 1 / (6 + self.count1[:, self.stimuli, self.action]))
			#self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) * 0.5 6

			self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
			
			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

		

	def reset(self):
		self.thalamus = np.zeros(self.context_num)
		self.thalamus[0] = 2 * self.a * self.tau
		self.ct = 0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))

		# self.prob = np.ones(( self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		
		self.stimuli_neurons = np.zeros(self.stimuli_num)
		self.value_neurons = np.zeros((self.context_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		
		self.prob =  0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num))
		self.pfc_neurons = np.zeros((self.stimuli_num, self.class_num, 2))
	


		self.count = np.zeros( (self.context_num, self.stimuli_num, self.class_num))
		self.count1 = np.zeros((self.context_num, self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0

		self.R = 0
		self.r = 0
		self.time = 0
		self.confidence = 0
		self.past_choice = [0]



		