import matplotlib.pyplot as plt
import matplotlib
from util import load_dict, save_dict, relu
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats
from matplotlib.collections import PolyCollection
import scikit_posthocs as sp
import pandas as pd
from posterior import * 
from sklearn.svm import LinearSVC 
import os
if not os.path.exists("./fig"):
    os.makedirs("./fig")
    
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


num = 1

sns.set_theme(context = "paper", style = "ticks")
#sns.set_theme("ticks", palette=None, context = "paper")
palette = sns.color_palette()
palette_alt = sns.color_palette("Set2")
palette_gradient = sns.color_palette("rocket")


def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

def plot(num):
	if num == 1:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		switch = len(data["switch"][agents[0]][0])
		max_trial = len(data["action"][agents[0]][0])
		max_episode = len(data["action"][agents[0]])
		print(max_episode)
		block_size = 25
		stimuli_num = 2
		class_num = 2
		context_num = 2
		scan_num = 20

		
		idx_df = ["None"] * 500
		
		pd.DataFrame(data["stimuli"][agents[0]]).to_csv("stimuli.csv")
		pd.DataFrame(data["action"][agents[0]]).to_csv("action.csv")
		pd.DataFrame(data["reward"][agents[0]] - np.concatenate([np.zeros((max_episode, 1)), data["reward"][agents[0]][:, :-1]], axis = 1)).to_csv("reward.csv")


		mb_idx = []
		mf_idx = []
		context_score = []
		for i in range(max_episode):
			md = np.vstack([ x["MD"]  for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, 2)[:, :, -1, :]
			df = md[i, 25:, 1] - md[i, 25:, 0]
			context_score.append(np.mean(df))
		

		


		stimulus = pd.read_csv("stimuli.csv").values[:, 1:]
		action = pd.read_csv("action.csv").values[:,1:]
		reward = pd.read_csv("reward.csv").values[:, 1:]

		model_free_score = []
		model_based_score = []
		mb_idx = []
		mf_idx = []

		for i in range(max_episode):
			model_free_score.append(model_free(stimulus[i, :], action[i, :], reward[i, :]))
			model_based_score.append(model_based(stimulus[i:i+1, :], action[i:i+1, :], reward[i:i+1, :]))

		score = np.array(model_free_score) - np.array(model_based_score)
		idx = np.argsort(score)
		
		mb_idx = np.arange(max_episode)[scipy.stats.zscore(context_score) > 0]
		mf_idx = np.arange(max_episode)[scipy.stats.zscore(context_score) < 0]

		for i in range(500):
			if i in mb_idx:
				idx_df[i] = "MB"
			else:
				idx_df[i] = "MF"

		pd.DataFrame(idx_df).to_csv("block.csv")

		# mb_idx = idx[:250]
		# mf_idx = idx[250:]
		print(len(mb_idx), len(mf_idx))

		idx_list = [mb_idx, mf_idx]
		idx_name = ["MB", "MF"]

		bt = np.arange(0, 50, 25)

		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		df = pd.DataFrame()
		df["x"] = scipy.stats.zscore(context_score)
		df["y"] = scipy.stats.zscore(score)
		sns.regplot(df, x = "x", y = "y", line_kws=dict(color="r"))
		res = scipy.stats.permutation_test((context_score, score), lambda x, y: scipy.stats.pearsonr(x, y).statistic, n_resamples=10000)
		print("{}, {}".format(res.statistic, res.pvalue))
		sns.despine()
		plt.savefig("fig/experiment{}_posterior_ratio.pdf".format(num), transparent = True)
		plt.close()

		


		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)



		for i, a in enumerate(idx_name):
			print(a)

			regret = np.mean(data["regret"][agents[0]][idx_list[i]], axis = 0)
			ci =  np.std(data["regret"][agents[0]][idx_list[i]], axis = 0) / np.sqrt(data["regret"][agents[0]][idx_list[i]].shape[0])
			
			plt.plot(t, regret, label = a, c = palette[i])
			plt.fill_between(t, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		
		plt.close()



	

		
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		t_a = np.arange(36) - 10

		for i, a in enumerate(idx_name):
			

			action = data["action"][agents[0]][idx_list[i]]
			stimuli = data["stimuli"][agents[0]][idx_list[i]]

			accuracy = np.zeros(action.shape)


			accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
			accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1

			accuracy_data = np.mean(accuracy[:, 14:50], axis = 0)
			ci =  np.std(accuracy[:, 14:50], axis = 0) / np.sqrt(accuracy[:, 15:50].shape[0])
			
			plt.plot(t_a, accuracy_data, label = a, c = palette[i])
			plt.fill_between(t_a, accuracy_data + ci, accuracy_data - ci, color = palette[i], alpha = 0.1)
		
		
		plt.axvline(0, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.axhline(0.5, c = "grey", linewidth = 1, linestyle = "dashed")

		ax.legend(loc = "lower left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
	
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()




		



		for i, a in enumerate(idx_name):
		

			evidence = np.mean(data["evidence"][agents[0]][idx_list[i]], axis = 0)
			ci =  np.std(data["evidence"][agents[0]][idx_list[i]], axis = 0) / np.sqrt(data["evidence"][agents[0]][idx_list[i]].shape[0])
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			plt.plot(t+1, evidence,  c =  palette[i])
			plt.fill_between(t + 1, evidence + ci, evidence - ci, color = palette[i], alpha = 0.1)
			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("Difference between two contextual populations\nfor {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_evidence_{}.pdf".format(num,a))
			plt.close()

			
			print(data["histogram"][agents[0]][0]["MD"].shape)
			md0 = np.vstack([ x["MD"][:, 0] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[idx_list[i], :, -1]
			md1 = np.vstack([ x["MD"][:, 1] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[idx_list[i], :, -1]


			md0_data = np.mean(md0, axis = 0)
			md1_data = np.mean(md1, axis = 0)
			ci0 =  np.std(md0, axis = 0) / np.sqrt(md0.shape[0])
			ci1 =  np.std(md1, axis = 0) / np.sqrt(md1.shape[0])

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			plt.plot(t[:40]-20, md0_data[5:45],  c = palette[i])
			plt.fill_between(t[:40] - 20, (md0_data + ci0)[5:45], (md0_data - ci0)[5:45], color = palette[0], alpha = 0.1)
			plt.plot(t[:40] - 20, md1_data[5:45],  c = palette[i])
			plt.fill_between(t[:40] - 20, (md1_data + ci1)[5:45], (md1_data - ci1)[5:45], color = palette[1], alpha = 0.1)
			
			plt.axvline(0, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("MD activities in {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_md_activities_{}.pdf".format(num,a))
			plt.close()

			
			ofc0 = np.vstack([ np.sum(x["BG"][:, 0, :], axis = -1) for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[idx_list[i], :, -1]
			
			ofc1 = np.vstack([ np.sum(x["BG"][:, 1, :], axis = -1) for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[idx_list[i], :, -1]


			ofc0_data = np.mean(ofc0, axis = 0)
			ofc1_data = np.mean(ofc1, axis = 0)
			ci0 =  np.std(ofc0, axis = 0) / np.sqrt(ofc0.shape[0])
			ci1 =  np.std(ofc1, axis = 0) / np.sqrt(ofc1.shape[0])

			
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			plt.plot(t[:40]-20, ofc0_data[5:45],  c = palette[i])
			plt.fill_between(t[:40] - 20, (ofc0_data + ci0)[5:45], (ofc0_data - ci0)[5:45], color = palette[0], alpha = 0.1)
			plt.plot(t[:40] - 20, ofc1_data[5:45],  c = palette[i])
			plt.fill_between(t[:40] - 20, (ofc1_data + ci1)[5:45], (ofc1_data - ci1)[5:45], color = palette[1], alpha = 0.1)
			
			plt.axvline(0, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("OFC activities in {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_ofc_activities_{}.pdf".format(num,a))
			plt.close()



		
			if i == 0:
				c = "purple"
			elif i == 1:
				c = "purple"
			elif i == 2:
				c = "green"
			else:
				c = "black"

			evidence = np.vstack([ x["MD"][:, 0] - x["MD"][:, 1] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[idx_list[i], :, -1]
			
			evidence[evidence <= 0] = -evidence[evidence <= 0]
			evidence = 2- 2 / (1 + np.exp(-evidence )) 
			evidence_data = np.mean(evidence, axis = 0)
			ci =  np.std(evidence, axis = 0) / np.sqrt(evidence.shape[0])




			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)


			plt.plot(t+1, evidence_data,  c = palette[i])
			plt.fill_between(t + 1, evidence_data + ci, evidence_data - ci, color = palette[i], alpha = 0.1)
				

			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.ylim(bottom = -0.05, top = 1)
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Probability density")
			plt.title("Contextual uncertainty for {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_uncertainty_{}.pdf".format(num,a))
			plt.close()

			
		
			md = np.vstack([ x["VIP"] - x["PV"] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, 2)[idx_list[i], :, -1, :]
			

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for j in range(2):
				md_lr = relu(2 /  (1 + np.exp( - 2*(md +0.25)))-1)
				md_lr_data = np.mean(md_lr, axis = 0)

				ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
				
				plt.plot(t[:40]-20, md_lr_data[:, j][5:45], label = "Context {}".format(j+1), c = palette[j])
				plt.fill_between(t[:40]-20, (md_lr_data[:, j] + ci[:, j])[5:45], (md_lr_data[:, j] - ci[:, j])[5:45], color = palette[j], alpha = 0.1)
			
			
			plt.axvline(0, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Learning rate")
			plt.title("Learning rate modulation from cortical interneurons over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_lr_in_{}.pdf".format(num,a))
			plt.close()

			
			md = np.vstack([ x["MD"] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, 2)[idx_list[i], :, -1, :]
		

			
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			for j in range(2):

				md_lr = relu(2 /  (1 + np.exp(8 - 4*(md-4)))- 1)
				md_lr_data = np.mean(md_lr, axis = 0)

				ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
				
				plt.plot(t + 1, md_lr_data[:, j], label = "Context {}".format(j+1), c = palette[j])
				plt.fill_between(t + 1, md_lr_data[:, j] + ci[:, j], md_lr_data[:, j] - ci[:, j], color = palette[j], alpha = 0.1)
			


			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left", frameon = False)
			plt.xlabel("Trial")
			plt.ylabel("Learning rate")
			plt.title("Learning rate of PFC-MD plasticity over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_lr_bcm_{}.pdf".format(num,a))
			plt.close()


		

		for i, a in enumerate(idx_name):
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			t = np.arange(max_trial)

			width = [1, 3]
			width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
			legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
	
			
			for k in range(2):
				for s in range(1):
					for j in range(2):

						value = np.vstack([ x["ALM/BG"] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, context_num, stimuli_num, class_num)[idx_list[i], :, -1, :, :, :]
		
						
						if j == 0:
							value_data = np.mean(value, axis = 0)[:, k, s, j]

							ci =  np.std(value[:, :, k, s, j], axis = 0) / np.sqrt(value[:, :, k, s, j].shape[0])
							print(ci.shape)
							ax.plot(t[:40]-20,value_data[5:45],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
							ax.fill_between(t[:40]-20, (value_data + ci)[5:45], (value_data - ci)[5:45], color = palette[i], alpha = 0.1)
						else:
							value_data = np.mean(value, axis = 0)[:, k, s, j]
							ci =  np.std(value[:, :, k, s, j], axis = 0) / np.sqrt(value[:, :, k, s, j].shape[0])
							ax.plot(t[:40]-20,  value_data[5:45], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
							ax.fill_between(t[:40]-20, (value_data + ci)[5:45], (value_data - ci)[5:45], color = palette[i], alpha = 0.1)
				
			line_patch = [mlines.Line2D([], [],color="grey", label=i, linestyle = t) for i, t in [("Left", "solid"), ("Right", "dashed")]]

	
			#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
			fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
			plt.axvline(0, c = "grey", linestyle = "dashed")
			plt.xlabel("Trial")
			plt.ylabel("Estimated value")
			plt.title("Estimated contextual value over {} trials".format(max_trial))
			plt.ylim(bottom = 0.35, top = 0.7)
			sns.despine()
			plt.savefig("fig/experiment{}_value_{}.pdf".format(num,a))
			plt.close()

		
		for i, a in enumerate(idx_name):
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			
			width = [1, 3]
			width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
			legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
	
			for k in range(2):
				for s in range(1):
					for j in range(2):
						model = np.vstack([ x["PFC/MD"] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, context_num, stimuli_num, class_num, 2)[idx_list[i], :, -1, :, :, :, 1]
		
						if j == 0:
							model_data = np.mean(model, axis = 0)[:, k, s, j]
							ci =  np.std(model[:, :, k, s, j], axis = 0) / np.sqrt(model[:, :, k, s, j].shape[0])
							ax.plot(t[:40]-20, model_data[5:45],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
							ax.fill_between(t[:40]-20, (model_data + ci)[5:45], (model_data - ci)[5:45], color = palette[i], alpha = 0.1)
						else:
							model_data = np.mean(model, axis = 0)[:, k, s, j]
							ci =  np.std(model[:, :, k, s, j], axis = 0) / np.sqrt(model[:, :, k, s, j].shape[0])
							ax.plot(t[:40]-20,  model_data[5:45], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
							ax.fill_between(t[:40]-20, (model_data + ci)[5:45], (model_data - ci)[5:45], color = palette[i], alpha = 0.1)
				

			fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
			plt.axvline(0, c = "grey", linestyle = "dashed")
			plt.xlabel("Trial")
			plt.ylabel("Estimated Probability")
			plt.title("Generative model of receiving reward over {} trials".format(max_trial))
			plt.ylim(bottom = 0.35, top = 0.7)
			sns.despine()
			plt.savefig("fig/experiment{}_model_{}.pdf".format(num, a))
			plt.close()

	
		

		

plot(num)




