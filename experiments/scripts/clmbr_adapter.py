import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from lightgbm import LGBMClassifier as gbm
from sklearn.metrics import log_loss

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.util import str2bool

from tune_adapter import (
	get_data,
	get_xy,
)

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
	description = "Train model on selected hyperparameters"
)

parser.add_argument(
	"--extracts_fpath",
	type = str,
	default = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/extracts/20220801',
	help = "path to extracts"
)

parser.add_argument(
	"--artifacts_fpath",
	type = str,
	default = '/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr',
	help = "path to clmbr artifacts including infos and models"
)

parser.add_argument(
	'--adapter_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr',
	help='Base path for the adapter model layer.'
)

parser.add_argument(
	'--results_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results/clmbr',
	help='Base path for the results.'
)

parser.add_argument(
	"--cohort_path",
	type = str,
	default = "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort",
	help = "path to save cohort"
)

parser.add_argument(
	"--n_jobs",
	type=int,
	default=4,
	help="number of threads"
)

parser.add_argument(
	"--seed",
	type = int,
	default = 44,
	help = "seed for deterministic training"
)

parser.add_argument(
	"--lr",
	type = str,
	default = "0.0001"
)

parser.add_argument(
	"--overwrite",
	type = str2bool,
	default = "false",
	help = "whether to overwrite existing artifacts",
)

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------
def get_model(C):
	return LogisticRegression(
					random_state = 44, 
					C = C, 
					max_iter = 10000,
					warm_start = True 
	)
def get_data(features_fpath):
	"""
	grab data
	"""

	features=pickle.load(gzip.open(os.path.join(features_fpath,"features.gz"),'rb'))
	prediction_ids=pickle.load(gzip.open(os.path.join(features_fpath,"prediction_ids.gz"),'rb'))
	labels=pickle.load(gzip.open(os.path.join(features_fpath,"labels.gz"),'rb'))
	ehr_ml_patient_ids=pickle.load(gzip.open(os.path.join(features_fpath,"ehr_ml_patient_ids.gz"),'rb'))
	day_indices=pickle.load(gzip.open(os.path.join(features_fpath,"day_indices.gz"),'rb'))

	return features,labels,prediction_ids,ehr_ml_patient_ids,day_indices


def get_xy(
	task,
	features,
	labels,
	prediction_ids,
	combine_train_val=False,
	get_test=False
	):
	if get_test:
		X_test=features[task]['test']
		y_test=labels[task]['test']
		prediction_id_tests=prediction_ids[task]['test']
		
		return (X_test, y_test, prediction_id_test)
	if combine_train_val:

		X_train=np.concatenate((
			features[task]['train'],
			features[task]['val']
		))

		y_train=np.concatenate((
			labels[task]['train'],
			labels[task]['val']
		))

		prediction_id_train=np.concatenate((
			prediction_ids[task]['train'],
			prediction_ids[task]['val']
		))

		return (X_train,y_train,prediction_id_train)
	else:
		X_train=features[task]['train']
		y_train=labels[task]['train']
		X_val=features[task]['val']
		y_val=labels[task]['val']
		prediction_id_train=prediction_ids[task]['train']
		prediction_id_val=prediction_ids[task]['val']


		return (X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val)
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

C = [1.0e-06,1.0e-05,0.0001,0.001,0.01,0.1,1]

# set seed
np.random.seed(args.seed)

# parse tasks and train_group
tasks =['hospital_mortality','sepsis','LOS_7','readmission_30','hyperkalemia_lab_mild_label','hyperkalemia_lab_moderate_label','hyperkalemia_lab_severe_label','hyperkalemia_lab_abnormal_label','hypoglycemia_lab_mild_label','hypoglycemia_lab_moderate_label','hypoglycemia_lab_severe_label','hypoglycemia_lab_abnormal_label','neutropenia_lab_mild_label','neutropenia_lab_moderate_label','neutropenia_lab_severe_label','hyponatremia_lab_mild_label','hyponatremia_lab_moderate_label','hyponatremia_lab_severe_label','hyponatremia_lab_abnormal_label','aki_lab_aki1_label','aki_lab_aki2_label','aki_lab_aki3_label','aki_lab_abnormal_label','anemia_lab_mild_label','anemia_lab_moderate_label','anemia_lab_severe_label','anemia_lab_abnormal_label','thrombocytopenia_lab_mild_label','thrombocytopenia_lab_moderate_label','thrombocytopenia_lab_severe_label','thrombocytopenia_lab_abnormal_label']

# initialize evaluator
evaluator = StandardEvaluator()

for cohort in ['all', 'ad']:
	print(f'Trained on cohort {cohort}')
	for train_type in ['pretrained', 'finetuned']:
		print(f'{train_type}')
		for task in tasks:
			adapter_save_path = f'{args.adapter_path}/adapters/{train_type}/{cohort}/{task}/tr_{"ped" if train_type == 'finetuned' else "ad"}_tst_ped/gru_sz_800_do_0_lr_{args.lr}_l2_0'
			result_save_path = f'{args.results_path}/{train_type}/{cohort}/{task}/tr_{"ped" if train_type == 'finetuned' else "ad"}_tst_ped/gru_sz_800_do_0_lr_{args.lr}_l2_0'
			os.makedirs(f"{adapter_save_path}",exist_ok=True)
			os.makedirs(f"{result_save_path}",exist_ok=True)
			print(f"task: {task}")

			train_feat_dir=os.path.join(
				args.artifacts_fpath,
				args.train_type,
				"features",
				args.cohort_id,
				f"gru_sz_800_do_0_lr_{args.lr}_l2_0",
				"ped" if train_type == 'finetuned' else "ad"
			)

			test_feat_dir=os.path.join(
				args.artifacts_fpath,
				args.train_type,
				"features",
				args.cohort_id,
				f"gru_sz_800_do_0_lr_{args.lr}_l2_0",
				"ped"
			)

			# get data
			tr_features,tr_labels,tr_prediction_ids,tr_ehr_ml_patient_ids,tr_day_indices = get_data(train_feat_dir)

			tst_features,tst_labels,tst_prediction_ids,tst_ehr_ml_patient_ids,tst_day_indices = get_data(test_feat_dir)

			# get data
			X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val = get_xy(
				task=task,
				features=tr_features,
				labels=tr_labels,
				prediction_ids=tr_prediction_ids,
				combine_train_val=False
			)
			  
			X_test,y_test,prediction_id_test = get_xy(
				task=task,
				features=tst_features,
				labels=tst_labels,
				prediction_ids=tst_prediction_ids,
				combine_train_val=False,
				get_test=True
			)

			best_loss = 9999999
			best_adapter = None
			for c in C:
				m = get_model(c)
				m.fit(X_train,y_train)
				val_preds = model.predict_proba(val_X)[:,1]

				loss = log_loss(y_val,val_preds)
				if loss < best_loss:
					best_loss = loss
					best_adapter = m
			
			df = pd.DataFrame()

			# save
			pickle.dump(
				best_adapter,
				open(f"{adapter_save_path}/model.pkl","wb")
			)


			pred_df = pd.DataFrame({
					'pred_probs':best_adapter.predict_proba(X_test)[:,1],
					'labels':y_test,
					'prediction_id':prediction_id_test,
					'task':task,
					'train_type':train_type,
					'pretrain_cohort':cohort,
					'phase':'test'
				})

			df_test = evaluator.evaluate(
				pred_df,
				strata_vars='phase',
				label_var='labels',
				pred_prob_var='pred_probs'
			)
			df_test['model'] = 'adapter'
			df_test['task'] = task
			df_test['train_type']:train_type
			df_test['pretrain_cohort']:cohort
			

			pred_df.reset_index(drop=True).to_csv(f"{result_save_path}/preds.csv")
			df_test.reset_index(drop=True).to_csv(f"{result_save_path}/test_eval.csv")