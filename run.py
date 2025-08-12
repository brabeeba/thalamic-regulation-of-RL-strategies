import pandas as pd
import scipy.io as sio
import glob
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from scipy.stats import pearsonr, ttest_1samp, beta
import util



matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_theme(context = "paper", style = "ticks")
#sns.set_theme("ticks", palette=None, context = "paper")
palette = sns.color_palette()
palette_alt = sns.color_palette("Set2")
palette_gradient = sns.color_palette("rocket")


filepath = glob.glob("./Logfile_RL_GoNoGO/*.txt")

data = sio.loadmat('Index_MF_MB.mat')
dataframes = {}
for file in filepath:
	with open(file, 'r') as f:
		header = f.readline().strip().split()  # Split header by spaces
	df = pd.read_csv(file, sep="\t", skiprows=1, names=header, index_col=False)

	dataframes[int(file[23:25])] = df

subject = list(dataframes.keys())
subject.sort()

flag = data["flag_block_MF_vs_MB_halves_from_dist"].T

variance = lambda x, y: x+y - 2


df = dataframes[3]
df['trial_diff'] = df['Trial_Nr'].diff().fillna(0)
df['segment_id'] = (df['trial_diff'] < 0).cumsum()

filter_df = df[(df['segment_id']== 6) ]
print(filter_df)
#print(((filter_df['Trial_type'] == 1).sum() + (filter_df['Trial_type'] == 3).sum()) / len(filter_df))

result = []
result_1 = []
result_2 = []
length = []
subject_list = []
for i, s in enumerate(subject):
	result_arr = []
	result_arr_1 = []
	result_arr_2 = []
	length_arr = []
	subject_arr = []

	for j in range(flag.shape[1]):

		subject_arr.append(i)

		alpha = np.ones((2, 2) ,dtype = np.int64)
		beta = np.ones((2, 2) ,dtype = np.int64)

		df = dataframes[s]
		df['trial_diff'] = df['Trial_Nr'].diff().fillna(0)
		df['segment_id'] = (df['trial_diff'] < 0).cumsum()
		filter_df = df[(df['segment_id']== j)]
		
		
		reward = []
		correct = []
		k = 0
		p = 0
		for _, row in filter_df.iterrows():

			trial_type = row['Trial_type']
			response_type = row['response_type']
			reversal = row['Reversal']

			remaining = filter_df.shape[0] - k
			k+= 1

			filter_bool = True
			
			if filter_bool:
				p+= 1

			if trial_type == 1:
				if response_type == 1:
					
					if filter_bool:
						alpha[0, 0] += 1
						reward.append(1)
						if ((j % 2 == 0) & (reversal == 0))or ((j % 2 == 1) & (reversal == 1)):
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 3:
					
					if filter_bool:
						beta[0, 1] += 1
						reward.append(0)
						if ((j % 2 == 0) & (reversal == 0)) or((j % 2 == 1) & (reversal == 1)):
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)
			if trial_type == 2:
				if response_type == 2:
					
					if filter_bool:
						beta[0, 0] += 1
						reward.append(0)
						if ((j % 2 == 0) & (reversal == 0)) or((j % 2 == 1) & (reversal == 1)):
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 4:
					
					if filter_bool:
						alpha[0, 1] += 1
						reward.append(1)
						if ((j % 2 == 0) & (reversal == 0)) or((j % 2 == 1) & (reversal == 1)):
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)

			if trial_type == 3:
				if response_type == 4:
					
					if filter_bool:
						alpha[1, 1] += 1
						reward.append(1)
						if ((j % 2 == 0) & (reversal == 0)) or((j % 2 == 1) & (reversal == 1)):
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 2:
					
					if filter_bool:
						beta[1, 0] += 1
						reward.append(0)
						if ((j % 2 == 0) & (reversal == 0))or ((j % 2 == 1) & (reversal == 1)):
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)
			if trial_type == 4:
				if response_type == 3:
					
					if filter_bool:
						beta[1, 1] += 1
						reward.append(0)
						if ((j % 2 == 0) & (reversal == 0)) or ((j % 2 == 1) & (reversal == 1)):
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 1:
					
					if filter_bool:
						alpha[1, 0] += 1
						reward.append(1) 
						if ((j % 2 == 0) & (reversal == 0)) or ((j % 2 == 1) & (reversal == 1)):
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)

		if ((j % 2 == 0) & (reversal == 0)) & ((j % 2 == 1) & (reversal == 1)):
			result_arr.append([variance(alpha[0, 0], beta[0, 0])  , variance(alpha[1, 1] , beta[1, 1]) ]  )
		else:
			result_arr.append([variance(alpha[0, 1] , beta[0, 1])  , variance(alpha[1, 0] , beta[1, 0])  ] )

		
		result_arr_1.append(reward)
		result_arr_2.append(correct)
		length_arr.append(p)



	result.append(result_arr)
	result_1.append(result_arr_1)
	result_2.append(result_arr_2)
	length.append(length_arr)
	subject_list.append(subject_arr)

length = np.array(length)

result = np.sum(np.array(result), axis = -1)
subject_list = np.array(subject_list)
result_2 = np.array(result_2)


data = {}
data["correct"] = result_2



result = []
result_1 = []
result_2 = []
length = []
subject_list = []
for i, s in enumerate(subject):
	result_arr = []
	result_arr_1 = []
	result_arr_2 = []
	length_arr = []
	subject_arr = []

	for j in range(flag.shape[1]):

		subject_arr.append(i)

		alpha = np.ones((2, 2) ,dtype = np.int64)
		beta = np.ones((2, 2) ,dtype = np.int64)

		df = dataframes[s]
		df['trial_diff'] = df['Trial_Nr'].diff().fillna(0)
		df['segment_id'] = (df['trial_diff'] < 0).cumsum()
		filter_df = df[(df['segment_id']== j) & (df['Reversal'] == 0)]
		
		
		reward = []
		correct = []
		k = 0
		p = 0
		for _, row in filter_df.iterrows():

			trial_type = row['Trial_type']
			response_type = row['response_type']

			remaining = filter_df.shape[0] - k
			k+= 1

			filter_bool = True
			
			if filter_bool:
				p+= 1

			if trial_type == 1:
				if response_type == 1:
					
					if filter_bool:
						alpha[0, 0] += 1
						reward.append(1)
						if j % 2 == 0:
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 3:
					
					if filter_bool:
						beta[0, 1] += 1
						reward.append(0)
						if j % 2 == 0:
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)
			if trial_type == 2:
				if response_type == 2:
					
					if filter_bool:
						beta[0, 0] += 1
						reward.append(0)
						if j % 2 == 0:
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 4:
					
					if filter_bool:
						alpha[0, 1] += 1
						reward.append(1)
						if j % 2 == 0:
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)

			if trial_type == 3:
				if response_type == 4:
					
					if filter_bool:
						alpha[1, 1] += 1
						reward.append(1)
						if j % 2 == 0:
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 2:
					
					if filter_bool:
						beta[1, 0] += 1
						reward.append(0)
						if j % 2 == 0:
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)
			if trial_type == 4:
				if response_type == 3:
					
					if filter_bool:
						beta[1, 1] += 1
						reward.append(0)
						if j % 2 == 0:
							correct.append(1)
						else:
							correct.append(0)
				elif response_type == 1:
					
					if filter_bool:
						alpha[1, 0] += 1
						reward.append(1) 
						if j % 2 == 0:
							correct.append(0)
						else:
							correct.append(1)
				else:
					print(trial_type, response_type)
					if filter_bool:
						reward.append(0)
						correct.append(0)

		if j % 2 == 0:
			result_arr.append([variance(alpha[0, 0], beta[0, 0])  , variance(alpha[1, 1] , beta[1, 1]) ]  )
		else:
			result_arr.append([variance(alpha[0, 1] , beta[0, 1])  , variance(alpha[1, 0] , beta[1, 0])  ] )

		
		result_arr_1.append(reward)
		result_arr_2.append(correct)
		length_arr.append(p)



	result.append(result_arr)
	result_1.append(result_arr_1)
	result_2.append(result_arr_2)
	length.append(length_arr)
	subject_list.append(subject_arr)

length = np.array(length)

result = np.sum(np.array(result), axis = -1)
subject_list = np.array(subject_list)


data["reversal"] = length
util.save_dict(data, "human_data")





# result_1 = np.array(result_1)
# result_2 = np.mean(np.array(result_2), axis = -1)


# print(np.mean(result_2[flag == 0, :]))
# print(np.mean(result_2[flag == 1, :]))



# for i in range(flag.shape[0]):

# 	ratio = 0.8
# 	fig, ax = plt.subplots()
# 	fig.set_figwidth(4.8 * ratio)
# 	fig.set_figheight(4.8 * ratio)

# 	ev = [result[i][flag[i] == 0], result[i][flag[i] == 1]]
# 	idx_name = ["MB", "MF"]
# 	ax.boxplot(ev, sym = '', widths = 0.7, showcaps = False, 
# 		                     vert=True,  # vertical box alignment
# 		                     labels=idx_name)  # will be used to label x-ticks
# 	z, p = scipy.stats.mannwhitneyu(ev[0], ev[1])
# 	if p < 0.05:
# 		print("hi", i)
# 	print("The p value of two-way rank sum test on the Context supporting evidence (SS) is {}".format(p))
# 	print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[0], np.mean(ev[0]), np.std(ev[0]) / np.sqrt(len(ev[0]))))
# 	print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[1],  np.mean(ev[1]), np.std(ev[1]) / np.sqrt(len(ev[1]))))

	
# 	sns.despine()
# 	plt.savefig("fig/human_subject_{}_value_belief_before_switch.pdf".format(i), transparent = True)
# 	plt.close()


# print(result_1[:5])
# print(result[5])
# print(flag[5])

# clf = LinearSVC()
# clf.fit(result_1.reshape(-1, 1), flag.flatten())
# print(clf.score(result_1.reshape(-1, 1), flag.flatten()))

# score = []

# for i, s in enumerate(subject): 
# 	clf = LinearSVC()

# 	clf.fit(result[i].reshape(-1, 1), flag[i])
# 	score.append(clf.score(result[i].reshape(-1, 1), flag[i]))
# print(score)
# print(np.mean(score))

# ratio = 0.8
# fig, ax = plt.subplots()
# fig.set_figwidth(4.8 * ratio)
# fig.set_figheight(4.8 * ratio)

# ev = [result[flag == 0], result[flag == 1]]

# df = pd.DataFrame()
# df['Group'] = flag.flatten()
# df['Value'] = result.flatten()
# df['Subject'] = subject_list.flatten()
# #sns.pointplot(x="Group", y="Value", data=df, ci=95, join=False, capsize=0.2, color='blue')

# # model = BinomialBayesMixedGLM.from_formula("Group ~ Value", {'Subject': '0 + C(Subject)'}, data=df)
# # result = model.fit_vb()

# # print(result.summary())


# idx_name = ["MB", "MF"]
# ax.boxplot(ev, sym = '', widths = 0.7, showcaps = False, 
# 	                      vert=True,  # vertical box alignment
# 	                      labels=idx_name)  # will be used to label x-ticks
# # Scatterplot with individual regression lines for each subject
# # for subject in df['Subject'].unique():
# #     sub_data = df[df['Subject'] == subject]
# #     plt.plot(sub_data['Value'], sub_data['Group'], marker='o', label=f"Subject {subject}", alpha=0.7)

# z, p = scipy.stats.mannwhitneyu(ev[0], ev[1], alternative='two-sided')
# print(z, p)
# print("The p value of two-way rank sum test on the Context supporting evidence (SS) is {} and z {}".format(p, z))
# print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[0], np.mean(ev[0]), np.std(ev[0]) / np.sqrt(len(ev[0]))))
# print("Context supporting evidence (SS) for {} = {}, sem = {}".format(idx_name[1],  np.mean(ev[1]), np.std(ev[1]) / np.sqrt(len(ev[1]))))
# sns.despine()
# plt.savefig("fig/human_subject_overall_value_belief_before_switch.pdf", transparent = True)
# plt.close()


# # Fixed effects
# intercept = 1.3268
# coef_value = -1.6878

# # Generate a range of Value
# x = np.linspace(0, 8, 100)
# log_odds = intercept + coef_value * x
# probabilities = np.exp(log_odds) / (1 + np.exp(log_odds))

# # Plot
# plt.plot(x, probabilities, label="Predicted Probability")
# plt.xlabel("Value")
# plt.ylabel("Probability of Outcome = 1")
# plt.title("Predicted Probability vs Value")
# plt.legend()
# plt.show()

# subject_correlations = []

# for subject in df['Subject'].unique():
#     sub_data = df[df['Subject'] == subject]
#     corr, _ = pearsonr(sub_data['Value'], sub_data['Group'])
#     subject_correlations.append(corr)

# # Perform a one-sample t-test
# t_stat, p_value = ttest_1samp(subject_correlations, 0)

# print(f"Mean correlation: {np.mean(subject_correlations)}, t-stat: {t_stat}, p-value: {p_value}")

# plt.hist(subject_correlations, bins=10, alpha=0.7, color='blue', edgecolor='black')
# plt.axvline(np.mean(subject_correlations), color='red', linestyle='dashed', linewidth=2, label='Mean Correlation')
# plt.title('Distribution of Subject-Level Correlations')
# plt.xlabel('Correlation')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()



#1 1 cue 1/Go reward
#1 3 cue 1/nogo noreward
#3 4 cue 2/nogo reward
#3 2 cue 2/go noreward

#2 2 cue 1/Go noreward
#2 4 cue 1/noGo reward
#4 3 cue 2/noGo noreward
#4 1 cue 2/Go reward

