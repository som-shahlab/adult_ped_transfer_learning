import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from .plot_utils import ece_calibration_curve

def bootstrap_noninferiority_test(
		treatment:pd.DataFrame, 
		control:pd.DataFrame,
		labels:pd.DataFrame,
		margin:float=0.05, 
		n_boot:int=1000,
		seed:int=444,
		higher_is_better:bool=True,
		metric:str='auroc',
	)->(float,float,list):
	"""

	"""
	# Assert that the treatment and control predictions are same length
	# If not take the overlapping prediction_ids
	try:
		assert(len(treatment)==len(control)==len(labels))
	except:
		# equalize rows
		if len(treatment)>len(control):
			treatment = treatment.query('prediction_id.isin(@control["prediction_id"].unique())')
			labels = labels.query('prediction_id.isin(@control["prediction_id"].unique())')
		else:
			control = control.query('prediction_id.isin(@treatment["prediction_id"].unique())')
			labels = labels.query('prediction_id.isin(@treatment["prediction_id"].unique())')
		assert(len(treatment)==len(control)==len(labels))

	# sort by prediction_id so all predictions have same index
	treatment = treatment.sort_values(by=['prediction_id']).reset_index(drop=True)
	control = control.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels = labels.sort_values(by=['prediction_id']).reset_index(drop=True)

	# get the overall performance for the control model
	control_full_scores = get_full_score(control, labels, metric)

	# bootstrap mean differences between models
	diffs, ts, cs = run_multibootstrap(treatment, control, labels, n_boot, seed, metric)

	# compute p value by % of differences that cross the margin
	if higher_is_better:
		p = np.sum(diffs<(-margin*np.mean(control_full_scores)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control_full_scores)))/len(diffs)
	else:
		p = np.sum(diffs>(-margin*np.mean(control_full_scores)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control_full_scores)))/len(diffs)

	# test ran as one-tailed, return 2*min(p, 1-p) to get two-tailed p-value
	return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs), ts, cs

def bootstrap_superiority_test(
		treatment:pd.DataFrame, 
		control:pd.DataFrame,
		labels:pd.DataFrame,
		n_boot:int=1000,
		seed:int=444,
		higher_is_better:bool=True,
		metric:str='auroc',
	)->(float,float,list):
	"""

	"""
	# Assert that the treatment and control predictions are same length
	# If not take the overlapping prediction_ids
	try:
		assert(len(treatment)==len(control)==len(labels))
	except:
		# equalize rows
		if len(treatment)>len(control):
			treatment = treatment.query('prediction_id.isin(@control["prediction_id"].unique())')
			labels = labels.query('prediction_id.isin(@control["prediction_id"].unique())')
		else:
			control = control.query('prediction_id.isin(@treatment["prediction_id"].unique())')
			labels = labels.query('prediction_id.isin(@treatment["prediction_id"].unique())')
		assert(len(treatment)==len(control)==len(labels))

	# sort by prediction_id so all predictions have same index
	treatment = treatment.sort_values(by=['prediction_id']).reset_index(drop=True)
	control = control.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels = labels.sort_values(by=['prediction_id']).reset_index(drop=True)

	# bootstrap mean differences between models
	diffs, ts, cs = run_multibootstrap(treatment, control, labels, n_boot, seed, metric)

	# compute p value by % of differences that cross 0
	if higher_is_better:
		p = np.sum(diffs<=0)/len(diffs)
	else:
		p = np.sum(diffs>=0)/len(diffs)

	# test ran as one-tailed, return 2*min(p, 1-p) to get two-tailed p-value
	return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs), ts, cs

def run_multibootstrap(
		treatment:pd.DataFrame, 
		control:pd.DataFrame,
		labels:pd.DataFrame,
		n_boot:int=1000,
		seed:int=444,
		metric:str='auroc',
	)->np.ndarray:
	np.random.seed(seed)
	diffs = np.empty(n_boot)
	diffs[:] = np.nan

	# get number of prediction samples and task samples
	n_samples = len(control['prediction_id'])
	tasks = control.drop('prediction_id', axis=1).columns
	n_tasks = len(tasks)

	for i in range(n_boot):
		# sample indices and tasks
		ids = control.sample(n=n_samples,replace=True).index
		sample_tasks = np.random.choice(tasks,n_tasks,replace=True)

		# slice prediction dfs using sampled indices
		treatment_df = treatment.iloc[ids]
		control_df = control.iloc[ids]
		labels_df = labels.iloc[ids]

		treatment_scores = []
		control_scores = []

		# compute sample metric score for each sampled task
		for task in sample_tasks:
			label_samples = labels_df[f'{task}_labels'].values
			label_samples = label_samples[~np.isnan(label_samples)] # removes patients that should be ignored for task
			treatment_samples = treatment_df[f'{task}'].values
			treatment_samples = treatment_samples[~np.isnan(treatment_samples)]  # removes patients that should be ignored for task
			control_samples = control_df[f'{task}'].values
			control_samples = control_samples[~np.isnan(control_samples)] # removes patients that should be ignored for task

			assert(len(label_samples)==len(treatment_samples)==len(control_samples))

			if metric == 'auroc':
				treatment_scores.append(roc_auc_score(label_samples, treatment_samples))
				control_scores.append(roc_auc_score(label_samples, control_samples))
			else:
				treatment_scores.append(ece_calibration_score(label_samples, treatment_samples))
				control_scores.append(ece_calibration_score(label_samples, control_samples))

		# compute the mean difference between models
		diffs[i] = np.mean(np.array(treatment_scores) - np.array(control_scores))

	return diffs, treatment_scores, control_scores

def get_full_score(
		df:pd.DataFrame, 
		labels:pd.DataFrame, 
		metric:str='auroc'
	)->np.ndarray:
	"""
	Computes full metric score for use in non-inferiority test calculation.
	"""
	tasks = df.drop('prediction_id', axis=1).columns
	scores = []

	for task in tasks:
		probs = df[task].values
		probs = probs[~np.isnan(probs)] # removes patients that should be ignored for task
		lbls = labels[f'{task}_labels'].values
		lbls = lbls[~np.isnan(lbls)] # removes patients that should be ignored for task

		assert(len(probs)==len(lbls))

		if metric == 'auroc':
			scores.append(roc_auc_score(lbls, probs))
		elif metric == 'ece':
			scores.append(ece_calibration_score(lbls, probs))

	return np.array(scores)

def bootstrap_noninferiority_test_robust(
	treatment:pd.DataFrame, 
	control:pd.DataFrame,
	labels:pd.DataFrame,
	margin:float=0.05, 
	n_boot:int=1000,
	seed:int=444,
	higher_is_better:bool=True,
	metric:str='auroc',
	)->(float,float,list):
	"""
	For use in ID vs OOD comparisons, as multibootstrap requires both models to be evaluated on the same test set.
	"""
	np.random.seed(seed)
	diffs = np.empty(n_boot)
	diffs[:] = np.nan
	# Assert that the treatment and control predictions are same length
	# If not take the overlapping prediction_ids
	try:
		assert(len(treatment)==len(labels_id))
	except:
		if len(treatment)>len(labels_id):
			treatment = treatment.query('prediction_id.isin(@labels_id["prediction_id"].unique())')
		else:
			labels_id = labels_id.query('prediction_id.isin(@treatment["prediction_id"].unique())')

	try:
		assert(len(control)==len(labels_ood))
	except:
		if len(control)>len(labels_ood):
			control = control.query('prediction_id.isin(@labels_ood["prediction_id"].unique())')
		else:
			labels_ood = labels_ood.query('prediction_id.isin(@control["prediction_id"].unique())')

	treatment = treatment.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels_id = labels_id.sort_values(by=['prediction_id']).reset_index(drop=True)

	control = control.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels_ood = labels_ood.sort_values(by=['prediction_id']).reset_index(drop=True)

	# get the overall performance for the control model
	control_full_scores = get_full_score(control, labels_ood, metric)
	treatment_full_scores = get_full_score(treatment, labels_id, metric)

	for i in range(n_boot):
		ids = np.random.choice(len(treatment_full_scores),len(treatment_full_scores),replace=True)
		diffs[i] = np.mean(treatment_full_scores[ids] - control_full_scores[ids])

	if higher_is_better:
		p = np.sum(diffs<(-margin*np.mean(control_full_scores)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control_full_scores)))/len(diffs)
	else:
		p = np.sum(diffs>(-margin*np.mean(control_full_scores)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control_full_scores)))/len(diffs)

	return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs)

def bootstrap_superiority_test_robust(
	treatment:pd.DataFrame, 
	control:pd.DataFrame,
	labels_id:pd.DataFrame,
	labels_ood:pd.DataFrame,
	n_boot:int=10000,
	seed:int=444,
	higher_is_better:bool=True,
	metric:str='auroc',
	)->(float,float,list):
	"""
	For use in ID vs OOD comparisons, as multibootstrap requires both models to be evaluated on the same test set.
	"""
	np.random.seed(seed)
	diffs = np.empty(n_boot)
	diffs[:] = np.nan

	try:
		assert(len(treatment)==len(labels_id))
	except:
		if len(treatment)>len(labels_id):
			treatment = treatment.query('prediction_id.isin(@labels_id["prediction_id"].unique())')
		else:
			labels_id = labels_id.query('prediction_id.isin(@treatment["prediction_id"].unique())')

	try:
		assert(len(control)==len(labels_ood))
	except:
		if len(control)>len(labels_ood):
			control = control.query('prediction_id.isin(@labels_ood["prediction_id"].unique())')
		else:
			labels_ood = labels_ood.query('prediction_id.isin(@control["prediction_id"].unique())')

	treatment = treatment.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels_id = labels_id.sort_values(by=['prediction_id']).reset_index(drop=True)

	control = control.sort_values(by=['prediction_id']).reset_index(drop=True)
	labels_ood = labels_ood.sort_values(by=['prediction_id']).reset_index(drop=True)

	# get the overall performance for the control model
	control_full_scores = get_full_score(control, labels_ood, metric)
	treatment_full_scores = get_full_score(treatment, labels_id, metric)

	for i in range(n_boot):
		ids = np.random.choice(len(treatment_full_scores),len(treatment_full_scores),replace=True)
		diffs[i] = np.mean(treatment_full_scores[ids] - control_full_scores[ids])

	if higher_is_better:
		p = np.sum(diffs<=0)/len(diffs)
	else:
		p = np.sum(diffs>=0)/len(diffs)

	return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs)