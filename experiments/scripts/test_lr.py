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

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid
#from lightgbm import LGBMClassifier as gbm

from prediction_utils.util import str2bool
from prediction_utils.model_evaluation import StandardEvaluator

#------------------------------------
# Arg parser
#------------------------------------
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
	'--tasks',
	type=str,
	default=['hospital_mortality','LOS_7','readmission_30','icu_admission','aki1_label','aki2_label','hg_label','np_500_label','np_1000_label']
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


#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

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
	
	return test_feats.todense(), cohort

def get_labels(args, task, cohort):
	return cohort.query(f'test_row_idx>=0 and {task}_fold_id!="ignore"')[['prediction_id', 'test_row_idx', f'{task}']]

def slice_sparse_matrix(mat, rows):
	'''
	Slice rows in sparse matrix using given rows indices
	'''
	mask = np.zeros(mat.shape[0], dtype=bool)
	mask[rows] = True
	w = np.flatnonzero(mask)
	sliced = mat[w,:]
	return sliced

def eval_model(args, task, model_path, result_path, X_test, y_test):
	m = pickle.load(open(f'{model_path}/model.pkl', 'rb'))

	evaluator = StandardEvaluator()
	df = pd.DataFrame({
		'pred_probs':m.predict_proba(X_test)[:,1],
		'labels':y_test.flatten(),
		'task':task,
		'test_group':'test',
		'prediction_id':test_pred_ids
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
	os.makedirs(f"results_save_fpath/{hp['C']}",exist_ok=True)

	df_test['C'] = hp['C']
	df_test['model'] = args.model
	df_test['CLMBR_model'] = 'PT' if cl_hp is None else 'CL'
	df_test_ci.reset_index(drop=True).to_csv(
		f"{results_save_path}/test_eval.csv"
	)


#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
np.random.seed(args.seed)

# parse tasks and train_group
tasks = args.tasks

test_data, cohort = load_data(args)

for task in tasks:
    
    print(f"task: {task}")
	
	test_labels= get_labels(args, task, cohort)
	test_X = slice_sparse_matrix(test_data, list(test_labels['test_row_idx']))
    for cohort_type in ['pediatric', 'adult', 'shared']:
		print(f"cohort type: {cohort_type}")
		for feat_group in ['pediatric', 'shared']:
			print(f"feature set: {feat_group}")
			model_path = f'{args.model_path}/{cohort_type}/lr/{task}/{feat_group}_feats/best'
			result_path = f'{args.result_path}/{cohort_type}/lr/{task}/{feat_group}_feats/best'
			eval_model(args, task, model_path, result_save_path, test_x, test_labels)
			
			
    
        
        
        

  