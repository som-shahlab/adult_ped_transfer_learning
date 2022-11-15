import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml
import time

import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.sparse import csr_matrix as csr
from sklearn.model_selection import ParameterGrid
from lightgbm import LGBMClassifier as gbm

from prediction_utils.util import str2bool
from prediction_utils.pytorch_utils.metrics import StandardEvaluator

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'--bin_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bin_features'
)

parser.add_argument(
	'--cohort_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort'
)

parser.add_argument(
	'--hparam_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/hyperparams'
)

parser.add_argument(
	'--model_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models'
)

parser.add_argument(
	'--result_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results'
)

parser.add_argument(
	'--cohort_type',
	type=str,
	default='pediatric'
)

parser.add_argument(
	'--task',
	type=str,
	default='hospital_mortality'
)

parser.add_argument(
	'--feat_group',
	type=str,
	default='shared'
)

parser.add_argument(
	'--verbose',
	type=int,
	default=0
)

parser.add_argument(
	'--n_jobs',
	type=int,
	default=8
)

parser.add_argument(
	'--n_boot',
	type=int,
	default=1000
)

parser.add_argument(
	'--seed',
	type=int,
	default='26'
)


#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

def read_file(filename, columns=None, **kwargs):
	'''
	Helper function to read parquet and csv files into DataFrame
	'''
	print(filename)
	load_extension = os.path.splitext(filename)[-1]
	if load_extension == ".parquet":
		return pd.read_parquet(filename, columns=columns,**kwargs)
	elif load_extension == ".csv":
		return pd.read_csv(filename, usecols=columns, **kwargs)

def get_model_hp(model_path):
	return yaml.safe_load(open(f"{model_path}/hp.yml"))

def load_data(args):

	cohort = read_file(
			os.path.join(
				args.cohort_path,
				"cohort_split.parquet"
			),
			engine='pyarrow'
		)

	fn = f'{args.bin_path}/{args.cohort_type}'

	test_feats = sp.load_npz(f'{fn}/test/{args.feat_group}_feats.npz')
	test_rows = pd.read_csv(f'{fn}/test/test_pred_id_map.csv')
	
	cohort = cohort.merge(test_rows, how='left', on='prediction_id')
	cohort['test_row_idx'] = cohort['test_row_idx'].fillna(-1).astype(int)

	return test_feats, cohort

def get_labels(args, task, cohort):
	test_cohort = cohort.query(f'test_row_idx>=0 and {task}_fold_id!="ignore"').sort_values(by='test_row_idx') 
	return test_cohort[['prediction_id', 'test_row_idx', f'{task}']]

def eval_model(args, task, model_path, result_path, X_test, y_test, hp):
	m = pickle.load(open(f'{model_path}/model.pkl', 'rb'))
	evaluator = StandardEvaluator(metrics=['auc','auprc','auprc_c','loss_bce','ace_abs_logistic_logit'])

	df = pd.DataFrame({
		'pred_probs':m.predict_proba(X_test)[:,1],
		'labels':np.array(list(y_test[args.task].values)).astype(np.float32),
		'task':task,
		'test_group':'test',
		'prediction_id':list(test_labels['prediction_id'].values)
	})
	
	df_test_ci, df_test = evaluator.bootstrap_evaluate(
		df,
		n_boot = args.n_boot,
		n_jobs = args.n_jobs,
		strata_vars_eval=['test_group'],
		strata_vars_boot=['labels'],
		patient_id_var='prediction_id',
		return_result_df = True
	)
	
	df_test['lr'] = hp['learning_rate']
	df_test['num_leaves'] = hp['num_leaves']
	df_test['boosting_type'] = hp['boosting_type']
	df_test['model'] = 'gbm'
	os.makedirs(f"{result_path}", exist_ok=True)
	df_test_ci.reset_index(drop=True).to_csv(f"{result_path}/test_eval.csv", index=False)


#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
np.random.seed(args.seed)

# parse tasks and train_group
task = args.task

test_data, cohort = load_data(args)


print(f"task: {task}")

test_labels= get_labels(args, task, cohort)
test_X = test_data[list(test_labels['test_row_idx'])].astype(np.float32)
for cohort_type in ['pediatric', 'adult']:
	print(f"cohort type: {cohort_type}")
	for feat_group in ['pediatric', 'shared', 'adult']:
		print(f"feature set: {feat_group}")
		model_path = f'{args.model_path}/{cohort_type}/gbm/{task}/{feat_group}_feats/best'
		hp = get_model_hp(model_path)
		print(hp)
		result_path = f'{args.result_path}/{cohort_type}/gbm/{task}/{feat_group}_feats/best'
		eval_model(args, task, model_path, result_path, test_X, test_labels, hp)







