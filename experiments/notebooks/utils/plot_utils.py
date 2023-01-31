import numpy as np
import pandas as pd
import os

from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.utils import (
	column_or_1d,
	indexable,
	check_matplotlib_support,
)
from sklearn.utils.validation import (
	_check_fit_params,
	_check_sample_weight,
	_num_samples,
	check_consistent_length,
	check_is_fitted,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
	"""Create a radar chart with `num_vars` axes.

	This function creates a RadarAxes projection and registers it.

	Parameters
	----------
	num_vars : int
		Number of variables for radar chart.
	frame : {'circle' | 'polygon'}
		Shape of frame surrounding axes.

	"""
	# calculate evenly-spaced axis angles
	theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

	class RadarAxes(PolarAxes):

		name = 'radar'

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			# rotate plot such that the first axis is at the top
			self.set_theta_zero_location('N')

		def fill(self, *args, closed=True, **kwargs):
			"""Override fill so that line is closed by default"""
			return super().fill(closed=closed, *args, **kwargs)

		def plot(self, *args, **kwargs):
			"""Override plot so that line is closed by default"""
			lines = super().plot(*args, **kwargs)
			for line in lines:
				self._close_line(line)

		def scatter(self, *args, **kwargs):
			super().scatter(*args,**kwargs)

		def _close_line(self, line):
			x, y = line.get_data()
			# FIXME: markers at x[0], y[0] get doubled-up
			if x[0] != x[-1]:
				x = np.concatenate((x, [x[0]]))
				y = np.concatenate((y, [y[0]]))
				line.set_data(x, y)

		def set_varlabels(self, labels, **kwargs):
			self.set_thetagrids(np.degrees(theta), labels, **kwargs)

		def _gen_axes_patch(self):
			# The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
			# in axes coordinates.
			if frame == 'circle':
				return Circle((0.5, 0.5), 0.5)
			elif frame == 'polygon':
				return RegularPolygon((0.5, 0.5), num_vars,
									  radius=.5, edgecolor=None,zorder=0)
			else:
				raise ValueError("unknown value for 'frame': %s" % frame)

		def draw(self, renderer):
			""" Draw. If frame is polygon, make gridlines polygon-shaped """
			if frame == 'polygon':
				gridlines = self.yaxis.get_gridlines()
				for gl in gridlines:
					gl.get_path()._interpolation_steps = num_vars
			super().draw(renderer)


		def _gen_axes_spines(self):
			if frame == 'circle':
				return super()._gen_axes_spines()
			elif frame == 'polygon':
				# spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
				spine = Spine(axes=self,
							  spine_type='circle',
							  path=Path.unit_regular_polygon(num_vars))
				# unit_regular_polygon gives a polygon of radius 1 centered at
				# (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
				# 0.5) in axes coordinates.
				spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
									+ self.transAxes)


				return {'polar': spine}
			else:
				raise ValueError("unknown value for 'frame': %s" % frame)

	register_projection(RadarAxes)
	return theta


def load_results(sensitivity_analysis:bool=False):

	tasks = [
		'hospital_mortality', 'sepsis', 'LOS_7','readmission_30','aki_lab_aki3_label',
		'hyperkalemia_lab_severe_label','hypoglycemia_lab_severe_label','hyponatremia_lab_severe_label',
		'neutropenia_lab_severe_label','anemia_lab_severe_label','thrombocytopenia_lab_severe_label'
	]

	if sensitivity_analysis:
		tasks = [
			'aki_lab_aki1_label',
			'hyperkalemia_lab_mild_label','hypoglycemia_lab_mild_label','hyponatremia_lab_mild_label',
			'neutropenia_lab_mild_label','anemia_lab_mild_label','thrombocytopenia_lab_mild_label'
		]

	metrics = ['auc', 'auprc', 'ace_abs_logistic_logit']

	model_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models'
	results_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results'
	best_clmbr_params = {'all':'gru_sz_800_do_0_lr_0.0001_l2_0', 'mix':'gru_sz_800_do_0_lr_0.0001_l2_0'}

	feat_groups = ['shared']
	cohort_types = ['pediatric', 'adult']

	# Get baseline model
	lr_df = pd.DataFrame()
	for task in tasks:
		for trc in ['adult', 'pediatric']:
			for tstc in ['pediatric', 'adult']:
				for fg in feat_groups:
					df = pd.read_csv(f'{results_path}/lr/{task}/tr_{trc}_tst_{tstc}/shared_feats/best/test_eval.csv')
					df['task'] = task
					df['train_cohort'] = trc
					df['test_cohort'] = tstc
					df['cohort'] = trc
					df['model'] = 'LR'
					df['feature_group'] = fg
					lr_df = pd.concat([lr_df,df])
	lr_p_df = pd.DataFrame()
	for task in tasks:
		for trc in ['constrain']:
			for tstc in ['pediatric']:
				for p in list(range(5,100,5)):
					for fg in feat_groups:
						try:
							df = pd.read_csv(f'{results_path}/lr/{task}/tr_{trc}_tst_{tstc}/shared_feats_{p}/best/test_eval.csv')
							df['task'] = task
							df['percent'] = p
							df['train_cohort'] = trc
							df['test_cohort'] = tstc
							df['cohort'] = trc
							df['model'] = 'LR'
							df['feature_group'] = fg
							lr_p_df = pd.concat([lr_p_df,df])
						except:
							print(f'missing {p} {tstc} {task} eval')
	gbm_df = pd.DataFrame()
	for task in tasks:
		for trc in ['pediatric']:
			for tstc in ['pediatric']:
				for fg in feat_groups:
					df = pd.read_csv(f'{results_path}/gbm/{task}/tr_{trc}_tst_{tstc}/shared_feats/best/test_eval.csv')
					df['task'] = task
					df['train_cohort'] = trc
					df['test_cohort'] = tstc
					df['cohort'] = trc
					df['model'] = 'lightGBM'
					df['feature_group'] = fg
					gbm_df = pd.concat([gbm_df,df])
# 	print('PED CLMBR')
	ped_clmbr_df = pd.DataFrame()
	for task in tasks:
		for ct in ['ped']:
			for tr_cht in ['ped']:
				for tst_cht in ['ped']:
					for lr in ['0.0001']:
						try:
							df = pd.read_csv(f'{results_path}/clmbr/pretrained/{ct}/{task}/tr_{tr_cht}_tst_{tst_cht}/gru_sz_800_do_0_lr_{lr}_l2_0/test_eval.csv')
							df['task'] = task
							df['cohort'] = ct
							df['train_adapter'] = tr_cht
							df['test_adapter'] = tst_cht
							df['model'] = '$CLMBR_{Peds}$'
							ped_clmbr_df = pd.concat([ped_clmbr_df,df])
						except:
							print(f'missing {ct} {tst_cht} {task} eval')
	ad_clmbr_df = pd.DataFrame()
	for task in tasks:
		for ct in ['ad_no_ped']:
			for tr_cht in ['ad', 'ped']:
				for tst_cht in ['ad','ped']:
					for lr in ['0.0001']:
						try:
							df = pd.read_csv(f'{results_path}/clmbr/pretrained/{ct}/{task}/tr_{tr_cht}_tst_{tst_cht}/gru_sz_800_do_0_lr_{lr}_l2_0/test_eval.csv')
							df['task'] = task
							df['cohort'] = ct
							df['train_adapter'] = tr_cht
							df['test_adapter'] = tst_cht
							df['model'] = '$CLMBR_{Source[Adult]}$'
							ad_clmbr_df = pd.concat([ad_clmbr_df,df])
						except:
							print(f'missing {ct} {tst_cht} {task} eval')
	ad_ftp_clmbr_df = pd.DataFrame()
	for task in tasks:
		for ct in ['ad']:
			for tr_cht in ['ped']:
				for tst_cht in ['ped']:
					for lr in ['0.0001']:
						try:
							df = pd.read_csv(f'{results_path}/clmbr/finetuned/{ct}/{task}/tr_{tr_cht}_tst_{tst_cht}/gru_sz_800_do_0_lr_{lr}_l2_0/test_eval.csv')
							df['task'] = task
							df['cohort'] = ct
							df['train_adapter'] = tr_cht
							df['test_adapter'] = tst_cht
							df['model'] = '$CLMBR_{FT-Pre[Adult]}$'
							ad_ftp_clmbr_df = pd.concat([ad_ftp_clmbr_df,df])
						except:
							print(f'missing {ct} {tst_cht} {task} eval')
	mix_clmbr_df = pd.DataFrame()
	for task in tasks:
		for ct in ['mix']:
			for tr_cht in ['ad', 'ped']:
				for tst_cht in ['ad','ped']:
					for lr in ['0.0001']:
						try:
							df = pd.read_csv(f'{results_path}/clmbr/pretrained/{ct}/{task}/tr_{tr_cht}_tst_{tst_cht}/gru_sz_800_do_0_lr_{lr}_l2_0/test_eval.csv')
							df['task'] = task
							df['cohort'] = ct
							df['train_adapter'] = tr_cht
							df['test_adapter'] = tst_cht
							df['model'] = '$CLMBR_{Combined}$'
							mix_clmbr_df = pd.concat([mix_clmbr_df,df])
						except:
							print(f'missing {ct} {tst_cht} {task} eval')
	ped_clmbr_df = ped_clmbr_df.replace({'metric':{'ace_abs_logistic_log':'ace_abs_logistic_logit', 'ace_rmse_logistic_log':'ace_rmse_logistic_logit'}})
	mix_clmbr_df = mix_clmbr_df.replace({'metric':{'ace_abs_logistic_log':'ace_abs_logistic_logit', 'ace_rmse_logistic_log':'ace_rmse_logistic_logit'}})
	ad_clmbr_df = ad_clmbr_df.replace({'metric':{'ace_abs_logistic_log':'ace_abs_logistic_logit', 'ace_rmse_logistic_log':'ace_rmse_logistic_logit'}})
	ad_ftp_clmbr_df = ad_ftp_clmbr_df.replace({'metric':{'ace_abs_logistic_log':'ace_abs_logistic_logit', 'ace_rmse_logistic_log':'ace_rmse_logistic_logit'}})

	metrics = ['auc']
	scenarios = ['p1','p2', 's1', 's2', 'e1', 'e2','e3', 'e4', 'e5','e6']
	plot_dict = {}
	for s in scenarios:
		plot_dict[s] = {}
		for metric in metrics:
			plot_dict[s][metric] = pd.DataFrame()
			if s == 'p1': #CLMBR_Adults-LR_Adults VS Count-LR_Peds
				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				lr_target = lr_df.query("train_cohort=='pediatric' and metric==@metric and feature_group=='shared' and test_cohort=='pediatric'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':lr_target['performance'],'Tasks':lr_target['task'],'Model':'$Count-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))
			elif s == 'p2': #CLMBR_Adults-LR_Adults VS Count-GBM_Peds
				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				gbm_target = gbm_df.query("train_cohort=='pediatric' and metric==@metric and feature_group=='shared' and test_cohort=='pediatric'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':gbm_target['performance'],'Tasks':gbm_target['task'],'Model':'$Count-GBM_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 's1': # CLMBR_Peds-LR_Peds VS CLMBR_Adults-LR_Adults
				clmbr_target = ped_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_target['performance'],'Tasks':clmbr_target['task'],'Model':'$CLMBR_{Peds}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))


			elif s == 's2': # CLMBR_Adults-LR_Adults (ID), CLMBR_Adults-LR_Adults (OOD)
				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ad'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}[ID]$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e1': # CLMBR_Adults-LR_Peds VS CLMBR_Adults-LR_Adults 
				clmbr_source = ad_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped'and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e2': # CLMBR_FT-Pre-LR_Peds VS CLMBR_Adults-LR_Adults
				clmbr_ft = ad_ftp_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_ft['performance'],'Tasks':clmbr_ft['task'],'Model':'$CLMBR_{FT-Pre}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e3': # CLMBR_FT-Pre-LR_Peds VS CLMBR_Adults-LR_Peds
				clmbr_ft = ad_ftp_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_ft['performance'],'Tasks':clmbr_ft['task'],'Model':'$CLMBR_{FT-Pre}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped'and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e4': # CLMBR_FT-Pre-LR_Peds VS CLMBR_Combined-LR_Peds
				clmbr_ft = ad_ftp_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_ft['performance'],'Tasks':clmbr_ft['task'],'Model':'$CLMBR_{FT-Pre}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = mix_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='mix' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Combined}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e5': # CLMBR_Combined-LR_Peds VS CLMBR_Adults-LR_Peds
				clmbr_source = mix_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='mix' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Combined}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='ad_no_ped'and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

			elif s == 'e6': # CLMBR_Combined-LR_Peds VS CLMBR_Adults-LR_Adults
				clmbr_source = mix_clmbr_df.query("train_adapter=='ped' and metric==@metric and cohort=='mix' and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Combined}-LR_{Peds}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))

				clmbr_source = ad_clmbr_df.query("train_adapter=='ad' and metric==@metric and cohort=='ad_no_ped'and test_adapter=='ped'")[['performance', 'task']]
				df = pd.DataFrame({'Performance':clmbr_source['performance'],'Tasks':clmbr_source['task'],'Model':'$CLMBR_{Adults}-LR_{Adults}$'})
				plot_dict[s][metric] = pd.concat((plot_dict[s][metric],df))
			else:
				pass

	return plot_dict

def ece_calibration_curve(
	y_true,
	y_prob,
	*,
	pos_label=None,
	n_bins=5,
	strategy="uniform",
):
	y_true = column_or_1d(y_true)
	y_prob = column_or_1d(y_prob)
	check_consistent_length(y_true, y_prob)
	pos_label = _check_pos_label_consistency(pos_label, y_true)

	if y_prob.min() < 0 or y_prob.max() > 1:
		raise ValueError("y_prob has values outside [0, 1].")

	labels = np.unique(y_true)
	if len(labels) > 2:
		raise ValueError(
			f"Only binary classification is supported. Provided labels {labels}."
		)
	y_true = y_true == pos_label

	if strategy == "quantile":  # Determine bin edges by distribution of data
		quantiles = np.linspace(0, 1, n_bins + 1)
		bins = np.percentile(y_prob, quantiles * 100)
	elif strategy == "uniform":
		bins = np.linspace(0.0, 1.0, n_bins + 1)
	else:
		raise ValueError(
			"Invalid entry to 'strategy' input. Strategy "
			"must be either 'quantile' or 'uniform'."
		)

	binids = np.searchsorted(bins[1:-1], y_prob)

	bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
	bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
	bin_total = np.bincount(binids, minlength=len(bins))

	nonzero = bin_total != 0
	prob_true = bin_true[nonzero] / bin_total[nonzero]
	prob_pred = bin_sums[nonzero] / bin_total[nonzero]

	ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
	return prob_true, prob_pred, ece

def load_ece_results():

	model_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models'
	results_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results'
	best_clmbr_params = {'all':'gru_sz_800_do_0_lr_0.0001_l2_0', 'mix':'gru_sz_800_do_0_lr_0.0001_l2_0'}

	feat_groups = ['shared']
	cohort_types = ['pediatric', 'adult']

	tasks = [
		'hospital_mortality', 'sepsis', 'LOS_7','readmission_30','aki_lab_aki3_label',
		'hyperkalemia_lab_severe_label','hypoglycemia_lab_severe_label','hyponatremia_lab_severe_label',
		'neutropenia_lab_severe_label','anemia_lab_severe_label','thrombocytopenia_lab_severe_label'
	]

	plot_dict = {}
	plot_dict['p'] = pd.DataFrame()
	plot_dict['s'] = pd.DataFrame()
	plot_dict['e'] = pd.DataFrame()

	nbins=10

	for task in tasks:
		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ad",
				task,
				"tr_ad_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Adults}-LR_{Adults}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['p'] = pd.concat((plot_dict['p'],df))

		df_probs_count = pd.read_csv(
			os.path.join(
				results_path,
				f"lr",
				task,
				"tr_pediatric_tst_pediatric",
				"shared_feats",
				"best",
				"preds.csv"
			)
		).round(3).query('prediction_id!=-3161636188530946875')
		count_probs = df_probs_count[['labels','pred_probs']]
		count_y, count_x, count_ece = ece_calibration_curve(count_probs['labels'], count_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[count_ece], 'Model':['$Count-LR_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['p'] = pd.concat((plot_dict['p'],df))

		df_probs_gbm = pd.read_csv(
			os.path.join(
				results_path,
				f"gbm",
				task,
				"tr_pediatric_tst_pediatric",
				"shared_feats",
				"best",
				"preds.csv"
			)
		).round(3).query('prediction_id!=-3161636188530946875')
		gbm_probs = df_probs_gbm[['labels','pred_probs']]
		gbm_y, gbm_x, gbm_ece = ece_calibration_curve(gbm_probs['labels'], gbm_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[gbm_ece], 'Model':['$Count-GBM_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['p'] = pd.concat((plot_dict['p'],df))

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ped",
				task,
				"tr_ped_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Peds}-LR_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['s'] = pd.concat((plot_dict['s'],df))

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ad",
				task,
				"tr_ad_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Adults}-LR_{Adults}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['s'] = pd.concat((plot_dict['s'],df))  

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ad",
				task,
				"tr_ad_tst_ad",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Adults}-LR_{Adults}[ID]$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['s'] = pd.concat((plot_dict['s'],df))  

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ad",
				task,
				"tr_ped_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Adults}-LR_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['e'] = pd.concat((plot_dict['e'],df))

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"ad",
				task,
				"tr_ad_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Adults}-LR_{Adults}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['e'] = pd.concat((plot_dict['e'],df))  

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"finetuned",
				"ad",
				task,
				"tr_ped_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{FT-Pre}-LR_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['e'] = pd.concat((plot_dict['e'],df))

		df_probs_clmbr = pd.read_csv(
			os.path.join(
				results_path,
				f"clmbr",
				"pretrained",
				"mix",
				task,
				"tr_ped_tst_ped",
				"gru_sz_800_do_0_lr_0.0001_l2_0",
				"preds.csv"
			)
		).round(3)
		clmbr_probs = df_probs_clmbr[['labels','pred_probs']]
		clmbr_y, clmbr_x, clmbr_ece = ece_calibration_curve(clmbr_probs['labels'], clmbr_probs['pred_probs'], n_bins=nbins)
		df = pd.DataFrame({'Performance':[clmbr_ece], 'Model':['$CLMBR_{Combined}-LR_{Peds}$'], 'metric':['ece'], 'Tasks':[task]})
		plot_dict['e'] = pd.concat((plot_dict['e'],df))

	return plot_dict

def load_predictions(sensitivity_analysis=False):
	tasks = [
		'hospital_mortality', 'sepsis', 'LOS_7','readmission_30','aki_lab_aki3_label',
		'hyperkalemia_lab_severe_label','hypoglycemia_lab_severe_label','hyponatremia_lab_severe_label',
		'neutropenia_lab_severe_label','anemia_lab_severe_label','thrombocytopenia_lab_severe_label'
	]

	if sensitivity_analysis:
		tasks = [
			'aki_lab_aki1_label',
			'hyperkalemia_lab_mild_label','hypoglycemia_lab_mild_label','hyponatremia_lab_mild_label',
			'neutropenia_lab_mild_label','anemia_lab_mild_label','thrombocytopenia_lab_mild_label'
		]


	model_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models'
	results_path = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results'

	feat_groups = ['shared']
	cohort_types = ['pediatric', 'adult']
	pred_dict = {}

	label_dict = {'adult':None, 'pediatric':None}

	# Get baseline model
	pred_dict['lr'] = {'adult':{'adult':None, 'pediatric':None}, 'pediatric':{'adult':None, 'pediatric':None}}

	for trc in ['adult', 'pediatric']:
		for tstc in ['pediatric', 'adult']:
			lr_df = pd.DataFrame()
			for i, task in enumerate(tasks):
				df = pd.read_csv(f'{results_path}/lr/{task}/tr_{trc}_tst_{tstc}/shared_feats/best/preds.csv')[['prediction_id', 'pred_probs', 'labels']]
				df = df.rename(columns={'pred_probs':f'{task}', 'labels':f'{task}_labels'})
				if i == 0:
					lr_df = df[['prediction_id', f'{task}']]
					label_df = df[['prediction_id', f'{task}_labels']]
				else:
					lr_df = pd.merge(lr_df,df[['prediction_id', f'{task}']], on='prediction_id', how='outer')
					label_df = pd.merge(label_df,df[['prediction_id', f'{task}_labels']], on='prediction_id', how='outer')
			pred_dict['lr'][trc][tstc] = lr_df
			label_dict[tstc] = label_df


	pred_dict['gbm'] = {'adult':{'adult':None, 'pediatric':None}, 'pediatric':{'adult':None, 'pediatric':None}}            

	for trc in ['pediatric']:
		for tstc in ['pediatric']:
			for fg in feat_groups:
				lr_df = pd.DataFrame()
				for i, task in enumerate(tasks):
					df = pd.read_csv(f'{results_path}/gbm/{task}/tr_{trc}_tst_{tstc}/shared_feats/best/preds.csv')[['prediction_id', 'pred_probs']]
					df = df.rename(columns={'pred_probs':f'{task}'})
					if i == 0:
						gbm_df = df
					else:
						gbm_df = pd.merge(gbm_df,df, on='prediction_id', how='outer')
				pred_dict['gbm'][trc][tstc] = lr_df

	pred_dict['clmbr_ped'] = {'ad':{'ad':None, 'ped':None}, 'ped':{'ad':None, 'ad':None}}

	for trc in ['ped']:
		for tstc in ['ped']:
			for fg in feat_groups:
				lr_df = pd.DataFrame()
				for i, task in enumerate(tasks):
					df = pd.read_csv(f'{results_path}/clmbr/pretrained/ped/{task}/tr_{trc}_tst_{tstc}/gru_sz_800_do_0_lr_0.0001_l2_0/preds.csv')[['prediction_id', 'pred_probs']]
					df = df.rename(columns={'pred_probs':f'{task}'})
					if i == 0:
						clmbr_df = df
					else:
						clmbr_df = pd.merge(clmbr_df,df, on='prediction_id', how='outer')
				pred_dict['clmbr_ped'][trc][tstc] = clmbr_df

	pred_dict['clmbr_ad'] = {'ad':{'ad':None, 'ped':None}, 'ped':{'ad':None, 'ad':None}}

	for trc in ['ad','ped']:
		for tstc in ['ad','ped']:
			if trc == 'ped' and tstc == 'ad':
				continue
			for fg in feat_groups:
				lr_df = pd.DataFrame()
				for i, task in enumerate(tasks):
					df = pd.read_csv(f'{results_path}/clmbr/pretrained/ad_no_ped/{task}/tr_{trc}_tst_{tstc}/gru_sz_800_do_0_lr_0.0001_l2_0/preds.csv')[['prediction_id', 'pred_probs']]
					df = df.rename(columns={'pred_probs':f'{task}'})
					if i == 0:
						clmbr_df = df
					else:
						clmbr_df = pd.merge(clmbr_df,df, on='prediction_id', how='outer')
				pred_dict['clmbr_ad'][trc][tstc] = clmbr_df

	pred_dict['clmbr_ft'] = {'ad':{'ad':None, 'ped':None}, 'ped':{'ad':None, 'ad':None}}

	for trc in ['ped']:
		for tstc in ['ped']:
			for fg in feat_groups:
				lr_df = pd.DataFrame()
				for i, task in enumerate(tasks):
					df = pd.read_csv(f'{results_path}/clmbr/finetuned/ad_no_ped/{task}/tr_{trc}_tst_{tstc}/gru_sz_800_do_0_lr_0.0001_l2_0/preds.csv')[['prediction_id', 'pred_probs']]
					df = df.rename(columns={'pred_probs':f'{task}'})
					if i == 0:
						clmbr_df = df
					else:
						clmbr_df = pd.merge(clmbr_df,df, on='prediction_id', how='outer')
				pred_dict['clmbr_ft'][trc][tstc] = clmbr_df

	pred_dict['clmbr_mix'] = {'ad':{'ad':None, 'ped':None}, 'ped':{'ad':None, 'ad':None}}

	for trc in ['ped']:
		for tstc in ['ped']:
			for fg in feat_groups:
				lr_df = pd.DataFrame()
				for i, task in enumerate(tasks):
					df = pd.read_csv(f'{results_path}/clmbr/pretrained/mix/{task}/tr_{trc}_tst_{tstc}/gru_sz_800_do_0_lr_0.0001_l2_0/preds.csv')[['prediction_id', 'pred_probs']]
					df = df.rename(columns={'pred_probs':f'{task}'})
					if i == 0:
						clmbr_df = df
					else:
						clmbr_df = pd.merge(clmbr_df,df, on='prediction_id', how='outer')
				pred_dict['clmbr_mix'][trc][tstc] = clmbr_df
				
	return pred_dict