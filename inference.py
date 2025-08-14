from model import *
from task import *
from experiment import *
from torch.utils.tensorboard import SummaryWriter
import sys
import copy


logger = logging.getLogger('Bandit')
thismodule = sys.modules[__name__]


def create_experiment(opt):
	current_infer = getattr(thismodule, "experiment{}".format(opt['experiment_num']))
	return current_infer(opt)



def experiment1(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [TwoTimeScaleNeuralAgent(opt)]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]]))
	#task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer












def model_parameter(opt, model_num):
	opt = copy.deepcopy(opt)

	new_opt = {}

	if model_num == 1
	:
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 25
		new_opt["tau"] = 1
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 40
		new_opt["learning"] = True
		new_opt["nonlinear"] = True
		new_opt["inhibit"] = False
		new_opt["d2"] = False
		new_opt["rescue"] = False
		new_opt["quantile_num"] = 100
		new_opt["N"] = 500
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 3
		new_opt["d_interval"] = 1000
		new_opt["scan_interval"] = 50




	
	logger.info('Model parameter is the following:')
	for k, v in new_opt.items():
		opt[k] = v
		logger.info('{}: {}'.format(k, v))

	return opt

