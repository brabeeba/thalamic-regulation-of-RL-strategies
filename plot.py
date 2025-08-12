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
from sklearn.metrics import roc_curve, auc



from sklearn.svm import LinearSVC 
from sklearn.cluster import KMeans
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
		human_data = load_dict("human_data")
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

		context_score = np.array(context_score)
		

		


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

		kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(context_score.reshape(-1, 1))

		if kmeans.cluster_centers_[0][0] > kmeans.cluster_centers_[1][0]:
			mb_idx = np.arange(max_episode)[kmeans.labels_ == 0]
			mf_idx = np.arange(max_episode)[kmeans.labels_ == 1]
		else:
			mb_idx = np.arange(max_episode)[kmeans.labels_ == 1]
			mf_idx = np.arange(max_episode)[kmeans.labels_ == 0]

		# mb_idx = np.arange(max_episode)[scipy.stats.zscore(context_score) > 0]
		# mf_idx = np.arange(max_episode)[scipy.stats.zscore(context_score) < 0]

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

		idx_label = np.zeros(max_episode)
		idx_label[mf_idx] = 1


		evidence_for_current_context = []

		
		evidence = np.where((stimulus[:, :25] == action[:, :25]) & (reward[:, :25] == 1), 1, np.where((stimulus[:, :25] != action[:, :25]) & (reward[:, :25] == 0), 1, 0))


		action = data["action"][agents[0]]
		stimuli = data["stimuli"][agents[0]]

		accuracy = np.zeros(action.shape, dtype = np.int64)

		accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
		accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1
		learned_idx  = np.arange(max_episode)

		mask = ((np.mean(accuracy[:, 14:25], axis= -1) > 0.5)) & (np.mean(accuracy[:, 39:50], axis= -1) > 0.5)



		learned_idx = np.arange(max_episode)

		mb_idx = sorted(set(learned_idx).intersection(mb_idx))
		mf_idx = sorted(set(learned_idx).intersection(mf_idx))

		idx_list = [mb_idx, mf_idx]

		print(len(learned_idx), len(mb_idx), len(mf_idx))


		
	
		

		print(stimulus[0, :25])
		print(action[0, :25])
		print(reward[0, :25])
		print(evidence[0])

		
		evidence_for_current_context = np.mean(evidence, axis = 1)

		clf = LinearSVC()
		clf.fit(evidence, idx_label)
		print(clf.score(evidence, idx_label))
		print(clf.coef_)
		print(clf.intercept_)

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)


		fpr, tpr, thresholds = roc_curve(idx_label, -evidence_for_current_context)
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, color=palette[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
		plt.plot([0, 1], [0, 1], 'k--', label="Random guessing (AUC = 0.5)")
		plt.xlabel("False Positive Rate (FPR)")
		plt.ylabel("True Positive Rate (TPR)")
		plt.legend(loc="lower right")
		sns.despine()
		plt.savefig("fig/experiment{}_auc_curve.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		df = pd.DataFrame()
		df["x"] = scipy.stats.zscore(scipy.stats.zscore(context_score))
		df["y"] = evidence_for_current_context
		sns.regplot(df, x = "x", y = "y", line_kws=dict(color="r"))
		res = scipy.stats.permutation_test((context_score, evidence_for_current_context), lambda x, y: scipy.stats.pearsonr(x, y).statistic, n_resamples=10000)
		print("{}, {}".format(res.statistic, res.pvalue))

		plt.xlabel("MD activity tuned to new context (SW)")
		plt.ylabel("Context supporting evidence (SS)")
		plt.legend(
    [f"r = {res.statistic:.2f}, p = {res.pvalue:.2g}"],
    loc="best",
    frameon=True,
    fontsize="medium",
   
)
		sns.despine()
		plt.savefig("fig/experiment{}_inputs_thalamic.pdf".format(num), transparent = True)
		plt.close()

		
		


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		ev_mb = evidence_for_current_context[idx_list[0]]
		ev_mf = evidence_for_current_context[idx_list[1]]

		x_mb = np.ones_like(ev_mb) + np.random.uniform(-0.1, 0.1, size=len(ev_mb))
		x_mf = np.ones_like(ev_mf) * 2 + np.random.uniform(-0.1, 0.1, size=len(ev_mf))

		df = pd.DataFrame()
		df["x"] = evidence_for_current_context
		df["y"] = idx_label

		sns.histplot(df, x="x", hue="y", bins=20, kde=True, alpha=0.5)
		plt.xlabel("Context supporting evidence (SS)")
		sns.despine()
		plt.legend(["MF", "MB"])
		plt.savefig("fig/experiment{}_context_evidence_histogram.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		ev_mb = evidence_for_current_context[idx_list[0]]
		ev_mf = evidence_for_current_context[idx_list[1]]

		ev = [ev_mb, ev_mf]
		

		ax.boxplot(ev, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=idx_name)  # will be used to label x-ticks
		plt.ylabel("Context supporting evidence (SS)")
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_context_evidence_box.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(ev[0], ev[1])
		print("The p value of two-way rank sum test on the Context supporting evidence (SS) is {}".format(p))
		print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[0], np.mean(ev[0]), np.std(ev[0]) / np.sqrt(len(ev[0]))))
		print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[1],  np.mean(ev[1]), np.std(ev[1]) / np.sqrt(len(ev[1]))))








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


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		action = data["action"][agents[0]]
		stimuli = data["stimuli"][agents[0]]

		accuracy = np.zeros(action.shape, dtype = np.int64)

		accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
		accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1

		human_correct = human_data["correct"].reshape(32 * 12, -1)
		reversal = human_data["reversal"].flatten()

		# human_correct 	= np.array([human_correct[i, :] for i in range(len(reversal))])
		label = ["Model", "Data"]
	

		accuracy_data = [np.mean(accuracy[learned_idx, :], axis = -1), np.mean(human_correct, axis = -1).flatten()]

		

		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=label)  # will be used to label x-ticks
		
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_model_vs_human_full_accuracy.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
		print("The p value of two-way rank sum test on accuracy of the entire block is {}".format(p))
		print("accuracy of the entire block for {} = {}, sem = {}, data number = {}".format(label[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0])), len(accuracy_data[0])))
		print("accuracy of the entire block for {} = {}, sem = {}, data number = {}".format(label[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1])), len(accuracy_data[1])))



		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		action = data["action"][agents[0]]
		stimuli = data["stimuli"][agents[0]]

		accuracy = np.zeros(action.shape, dtype = np.int64)

		accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
		accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1

		human_correct = human_data["correct"].reshape(32 * 12, -1)
		reversal = human_data["reversal"].flatten()

		human_correct 	= np.array([human_correct[i, reversal[i] - 10:reversal[i]] for i in range(len(reversal))])
		print(human_correct.shape)
		label = ["Model", "Data"]

		accuracy_data = [np.mean(accuracy[learned_idx, 15:25], axis = -1), np.mean(human_correct, axis = -1)]
		

		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=label)  # will be used to label x-ticks
		
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_model_vs_human_SS_accuracy.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
		print("The p value of two-way rank sum test on accuracy of the SS is {}".format(p))
		print("accuracy of the SS for {} = {}, sem = {}, data number = {}".format(label[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0])), len(accuracy_data[0])))
		print("accuracy of the SS for {} = {}, sem = {}, data number = {}".format(label[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1])), len(accuracy_data[1])))

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		action = data["action"][agents[0]]
		stimuli = data["stimuli"][agents[0]]

		accuracy = np.zeros(action.shape, dtype = np.int64)

		accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
		accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1

		human_correct = human_data["correct"].reshape(32 * 12, -1)
		reversal = human_data["reversal"].flatten()

		human_correct 	= np.array([human_correct[i, reversal[i]:reversal[i] + 10] for i in range(len(reversal))])
		print(human_correct.shape)
		label = ["Model", "Data"]

		accuracy_data = [np.mean(accuracy[learned_idx, 25:35], axis = -1), np.mean(human_correct, axis = -1)]
		

		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=label)  # will be used to label x-ticks
		
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_model_vs_human_SW_accuracy.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
		print("The p value of two-way rank sum test on accuracy of the SW is {}".format(p))
		print("accuracy of the SW for {} = {}, sem = {}, data number = {}".format(label[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0])), len(accuracy_data[0])))
		print("accuracy of the SW for {} = {}, sem = {}, data number = {}".format(label[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1])), len(accuracy_data[1])))


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(6.4 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		t_a = np.arange(50) - 25

		
			

		action = data["action"][agents[0]]
		stimuli = data["stimuli"][agents[0]]

		accuracy = np.zeros(action.shape)


		accuracy[:, :25][(action[:, :25] - stimuli[:, :25]) == 0] = 1
		accuracy[:, 25:][(action[:, 25:] - stimuli[:, 25:]) != 0] = 1

		accuracy_data = np.mean(accuracy[learned_idx, :], axis = 0)
		ci =  np.std(accuracy[learned_idx, :], axis = 0) / np.sqrt(accuracy[:, :].shape[0])
		
		plt.plot(t_a, accuracy_data, c = "black")
		plt.fill_between(t_a, accuracy_data + ci, accuracy_data - ci, color = "black", alpha = 0.1)
		
		
		plt.axvline(0, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.axhline(0.5, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.ylim(bottom = -0.05, top = 1)

		
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
	
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action_overall.pdf".format(num), transparent = True)
		plt.close()

		accuracy_data = np.mean(accuracy[learned_idx, :], axis = 0)
		ci =  np.std(accuracy[learned_idx, :], axis = 0) / np.sqrt(accuracy.shape[0])



		print("Accuracy before reversal is {} with sem {} and at the end of the trial is {} with sem {}".format(accuracy_data[24], ci[24], accuracy_data[-1], ci[-1]))
















		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		
		md1 = np.vstack([ np.maximum(x["MD"][:, 1], 0) for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num)[:, :, -1]

		context_data = np.zeros(md1.shape)
		context_data[:, 25:] = 1
		context_label = ["Context 1", "Context 2"]

		md_data = [md1[context_data == 0], md1[context_data == 1]]
		

		ax.boxplot(md_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=context_label)  # will be used to label x-ticks
		plt.ylabel("MD activity tuned to new context")
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_md_context_tuning.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(md_data[0], md_data[1])
		print("The p value of two-way rank sum test on the MD activity tuned to new context is {}".format(p))
		print("MD activity tuned to new context for {} = {}, sem = {}".format(context_label[0], np.mean(md_data[0]), np.std(md_data[0]) / np.sqrt(len(md_data[0]))))
		print("MD activity tuned to new context for {} = {}, sem = {}".format(context_label[1],  np.mean(md_data[1]), np.std(md_data[1]) / np.sqrt(len(md_data[1]))))


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

	
		model = np.vstack([ x["PFC/MD"] for x in data["histogram"][agents[0]]]).reshape(max_episode, max_trial, scan_num, context_num, stimuli_num, class_num, 2)[:, 24, -1, :, :, :, 1]

		model_diff1 = model[:, 0, 0, 0] - model[:, 0, 0, 1]
		model_diff2 = model[:, 0, 1, 1] - model[:, 0, 1, 0]
		
		diff_data = [ np.concatenate([model_diff1[idx_list[1]], model_diff2[idx_list[1]]]), np.concatenate([model_diff1[idx_list[0]], model_diff2[idx_list[0]]])]

		ax.boxplot(diff_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=[idx_name[1], idx_name[0]])  # will be used to label x-ticks
		plt.ylabel("Value (Correct - Incorrect)")
		plt.tight_layout()
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_correct-incorrect_value.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(diff_data[0], diff_data[1])
		print("The p value of two-way rank sum test on the Value (Go - NoGo | Cue 1 is {}".format(p))
		print("Value (Go - NoGo | Cue 1 for {} = {}, sem = {}".format(idx_name[1], np.mean(diff_data[0]), np.std(diff_data[0]) / np.sqrt(len(diff_data[0]))))
		print("Value (Go - NoGo | Cue 1 for {} = {}, sem = {}".format(idx_name[0],  np.mean(diff_data[1]), np.std(diff_data[1]) / np.sqrt(len(diff_data[1]))))


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		l2_distance = (model[:, 0, 0, 0] - 0.7)**2 + (model[:, 0, 0, 1] - 0.3)**2 + (model[:, 0, 1, 0] - 0.3)**2 + (model[:, 0, 1, 1] - 0.7)**2
		l2_distance = np.sqrt(l2_distance)
		
		l2_data = [ l2_distance[idx_list[1]], l2_distance[idx_list[0]] ]

		ax.boxplot(l2_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=[idx_name[1], idx_name[0]])  # will be used to label x-ticks
		plt.ylabel("L2 distance")
		plt.tight_layout()
		sns.despine()
		plt.legend(loc="upper left", frameon=False)
		plt.savefig("fig/experiment{}_l2_distance.pdf".format(num), transparent = True)
		plt.close()

		z, p = scipy.stats.mannwhitneyu(l2_data[0], l2_data[1])
		print("The p value of two-way rank sum test on the l2 distance is {}".format(p))
		print("l2 distance for {} = {}, sem = {}".format(idx_name[1], np.mean(l2_data[0]), np.std(l2_data[0]) / np.sqrt(len(l2_data[0]))))
		print("l2 distance for {} = {}, sem = {}".format(idx_name[0],  np.mean(l2_data[1]), np.std(l2_data[1]) / np.sqrt(len(l2_data[1]))))



		







		



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

# data = {}

# data_1 = load_dict("experiment33_data_2")
# data_2 = load_dict("experiment33_data_1")

# # #data_3 = load_dict("experiment11_data_4")
# keys = data_2.keys()
# agents = data_1["action"].keys()

# for k in keys:


# 	if k == "task":

# 		data[k] = np.vstack([data_1[k], data_2[k]])

		
# 	else:
# 		data[k] = {}
# 		for a in agents:

			
# 			if (k == "evidence" or k == "value" or k== "model") and (a == "Bayesian RL" or a == "Discounted Thompson Sampling"):
# 				continue

# 			elif (k == "quantile_data") and (a == "Discounted Thompson Sampling" or a == "HMM Model"):
# 				continue
# 			elif k== "scalars" or k=="histogram":
# 				data_1[k][a].extend(data_2[k][a])
# 				data[k][a] = data_1[k][a]
# 				print(len(data_1[k][a]))
# 			else:
# 				data[k][a] = np.vstack([data_1[k][a], data_2[k][a]])
# 				#data[k][a] =  data_2[k][a]


# save_dict(data, "experiment33_data")
			








