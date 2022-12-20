import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml
import gzip

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import log_loss

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
	description = "Train model on selected hyperparameters"
)

parser.add_argument(
	"--extracts_fpath",
	type = str,
	default = '/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/extracts/20220801',
	help = "path to extracts"
)

parser.add_argument(
	"--artifacts_fpath",
	type = str,
	default = '/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr',
	help = "path to clmbr artifacts including infos and models"
)

parser.add_argument(
	'--adapter_path',
	type=str,
	default='/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr',
	help='Base path for the adapter model layer.'
)

parser.add_argument(
	'--results_path',
	type=str,
	default='/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/results/clmbr',
	help='Base path for the results.'
)

parser.add_argument(
	"--cohort_path",
	type = str,
	default = "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/cohort",
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
	"--train_cohort",
	type = str,
	default = "ad"
)

parser.add_argument(
	"--eval_cohort",
	type = str,
	default = "ped"
)

parser.add_argument(
	"--constrain",
	type = str2bool,
	default = "false",
	help = "use constrained features",
)

parser.add_argument(
	"--percent",
	type = int,
	default = 5)

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
	return lr(
					random_state = 44, 
					C = C, 
					max_iter = 10000,
					warm_start = True 
	)

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

def get_data(features_fpath, ri=False):
	"""
	grab data
	"""

	features=pickle.load(gzip.open(os.path.join(features_fpath,"features.gz"),'rb'))
	prediction_ids=pickle.load(gzip.open(os.path.join(features_fpath,"prediction_ids.gz"),'rb'))
	labels=pickle.load(gzip.open(os.path.join(features_fpath,"labels.gz"),'rb'))
	ehr_ml_patient_ids=pickle.load(gzip.open(os.path.join(features_fpath,"ehr_ml_patient_ids.gz"),'rb'))
	day_indices=pickle.load(gzip.open(os.path.join(features_fpath,"day_indices.gz"),'rb'))

	if ri:
		row_indices=pickle.load(gzip.open(os.path.join(features_fpath,"row_indices.gz"),'rb'))

		return features,labels,prediction_ids,ehr_ml_patient_ids,day_indices, row_indices
	else:
		return features,labels,prediction_ids,ehr_ml_patient_ids,day_indices

def get_feat_idx(feat_pids, cohort_pids):
	# To save space on generated features all tasks that have an index date of midnight on admission share features.
	# Some tasks will have an ignore flag for certain train rows if the task is observed in patient timeline before
	# midnight of admission day. Therefore must get the indices of all train patients that are not ignored by task
	# using this function.
	return feat_pids[feat_pids.isin(cohort_pids.values)].index.values

def get_xy_constrain(task, features, labels, prediction_ids, cohort, cohort_group, row_indices, return_test=False):
	

	cohort = cohort.query('adult_at_admission==0')
	cohort = cohort.query('constrain==1 | fold_id=="test"')
	
	
	if return_test:
		prediction_id_test = prediction_ids[task]['test']
		X_test = features[task if task == 'readmission_30' else 'all']['test']
		y_test = np.array(labels[task]['test']).astype(np.int32)
		
		tst_idx = row_indices[task]['test']
	
		return (X_test[tst_idx],y_test,prediction_id_test)
	
	prediction_id_train=prediction_ids[task]['train']
	prediction_id_val=prediction_ids[task]['val']

	X_train = features[task if task == 'readmission_30' else 'all']['train']
	y_train = np.array(labels[task]['train']).astype(np.int32)
	
	X_val = features[task if task == 'readmission_30' else 'all']['val']
	y_val = np.array(labels[task]['val']).astype(np.int32)
	
	tr_idx = row_indices[task]['train']
	v_idx = row_indices[task]['val']

	return (X_train[tr_idx],y_train,prediction_id_train,X_val[v_idx],y_val,prediction_id_val)


def get_xy(task, features, labels, prediction_ids, cohort, cohort_group, return_test=False):
	
	if cohort_group == 'ped':
		cohort = cohort.query('adult_at_admission==0')
	else:
		cohort = cohort.query('adult_at_admission==1')
	
	if return_test:
		tst_c = cohort.query(f"{task}_fold_id==['test']")
		prediction_id_test = prediction_ids[task]['test']
		X_test = features[task if task == 'readmission_30' else 'all']['test']
		y_test = np.array(tst_c[task].values).astype(np.int32)

		if task == 'readmission_30':
			return (X_test,y_test,prediction_id_test)
		
		tst_idx = get_feat_idx(prediction_id_test, tst_c['prediction_id'])
		return (X_test[tst_idx],y_test,prediction_id_test[tst_idx])
	
	tr_c = cohort.query(f"{task}_fold_id!=['test','val','ignore']")
	v_c = cohort.query(f"{task}_fold_id==['val']")
	
	prediction_id_train=prediction_ids[task]['train']
	prediction_id_val=prediction_ids[task]['val']

	X_train = features[task if task == 'readmission_30' else 'all']['train']

	if task == 'readmission_30':
		y_train = np.array(labels[task]['train']).astype(np.int32)
	else:
		y_train = np.array(tr_c[task].values).astype(np.int32)

	X_val = features[task if task == 'readmission_30' else 'all']['val']
	
	if task == 'readmission_30':
		y_val = np.array(labels[task]['val']).astype(np.int32)
	else:
		y_val = np.array(v_c[task].values).astype(np.int32)
	
	if task == 'readmission_30':
		return (X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val)
	

	tr_idx = get_feat_idx(prediction_id_train, tr_c['prediction_id'])
	v_idx = get_feat_idx(prediction_id_val, v_c['prediction_id'])

	return (X_train[tr_idx],y_train,prediction_id_train[tr_idx],X_val[v_idx],y_val,prediction_id_val[v_idx])
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

C = [1.0e-06,1.0e-05,0.0001,0.001,0.01,0.1,1]

# set seed
np.random.seed(args.seed)

if args.constrain:
	cohort = read_file(
		os.path.join(
			args.cohort_path,
			f"cohort_split_no_nb_constrain_{args.percent}.parquet"
		),
		engine='pyarrow'
	)
else:
	cohort = read_file(
		os.path.join(
			args.cohort_path,
			"cohort_split_no_nb.parquet"
		),
		engine='pyarrow'
	)
# remove problematic patients that have timeline issues
cohort_df = cohort[~cohort['person_id'].isin([86281596,72463221, 31542622, 30046470])]
cohort_df = cohort_df.query('pediatric_age_group!="term neonatal"')
# parse tasks and train_group
tasks =['hospital_mortality','sepsis','LOS_7','readmission_30','hyperkalemia_lab_mild_label','hyperkalemia_lab_moderate_label','hyperkalemia_lab_severe_label','hyperkalemia_lab_abnormal_label','hypoglycemia_lab_mild_label','hypoglycemia_lab_moderate_label','hypoglycemia_lab_severe_label','hypoglycemia_lab_abnormal_label','neutropenia_lab_mild_label','neutropenia_lab_moderate_label','neutropenia_lab_severe_label','hyponatremia_lab_mild_label','hyponatremia_lab_moderate_label','hyponatremia_lab_severe_label','hyponatremia_lab_abnormal_label','aki_lab_aki1_label','aki_lab_aki2_label','aki_lab_aki3_label','aki_lab_abnormal_label','anemia_lab_mild_label','anemia_lab_moderate_label','anemia_lab_severe_label','anemia_lab_abnormal_label','thrombocytopenia_lab_mild_label','thrombocytopenia_lab_moderate_label','thrombocytopenia_lab_severe_label','thrombocytopenia_lab_abnormal_label']

# initialize evaluator
evaluator = StandardEvaluator()

for cohort in ['all', 'ad']:#['all', 'ad', 'ped']:
	print(f'Trained on cohort {cohort}')
	for train_type in ['pretrained', 'finetuned']:
		if train_type == 'pretrained' and args.constrain:
			continue
		train_feat_dir=os.path.join(
			args.artifacts_fpath,
			train_type,
			"features",
			cohort,
			f"gru_sz_800_do_0_lr_{args.lr}_l2_0",
			args.train_cohort if args.train_cohort != 'constrain' else f'constrain_{args.percent}'
		)

		test_feat_dir=os.path.join(
			args.artifacts_fpath,
			train_type,
			"features",
			cohort,
			f"gru_sz_800_do_0_lr_{args.lr}_l2_0",
			args.eval_cohort
		)
		# get data
		if args.train_cohort == 'constrain':
			tr_features,tr_labels,tr_prediction_ids,tr_ehr_ml_patient_ids,tr_day_indices, tr_row_indices = get_data(train_feat_dir, True)
		else:
			tr_features,tr_labels,tr_prediction_ids,tr_ehr_ml_patient_ids,tr_day_indices = get_data(train_feat_dir)
			
		tst_features,tst_labels,tst_prediction_ids,tst_ehr_ml_patient_ids,tst_day_indices = get_data(test_feat_dir)
		print(f'{train_type}')
		for task in tasks:
			adapter_save_path = f'{args.adapter_path}/adapters/{train_type}/{cohort}/{task}/tr_{args.train_cohort}_tst_{args.eval_cohort}/gru_sz_800_do_0_lr_{args.lr}_l2_0'
			result_save_path = f'{args.results_path}/{train_type}/{cohort}/{task}/tr_{args.train_cohort}_tst_{args.eval_cohort}/gru_sz_800_do_0_lr_{args.lr}_l2_0'
			os.makedirs(f"{adapter_save_path}",exist_ok=True)
			os.makedirs(f"{result_save_path}",exist_ok=True)
			print(f"task: {task}")
			# get data
			if args.train_cohort == 'constrain':
				X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val = get_xy_constrain(
					task,
					tr_features,
					tr_labels,
					tr_prediction_ids,
					cohort_df,
					args.train_cohort,
					tr_row_indices
				)
			else:
				X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val = get_xy(
					task,
					tr_features,
					tr_labels,
					tr_prediction_ids,
					cohort_df,
					args.train_cohort
				)
			X_test,y_test,prediction_id_test = get_xy(
				task,
				tst_features,
				tst_labels,
				tst_prediction_ids,
				cohort_df,
				args.eval_cohort,
				return_test=True
			)
			
			best_loss = np.inf
			best_adapter = None
			for c in C:
				print(f'Evaluating adapter with C={c}')
				m = get_model(c)
				m.fit(X_train,y_train)
				val_preds = m.predict_proba(X_val)[:,1]

				loss = log_loss(y_val,val_preds)
				print(f'val loss: {loss}')
				if loss < best_loss:
					print('saving best model...')
					best_loss = loss
					best_c = c
					best_adapter = m
			
			df = pd.DataFrame()

			# save
			print('saving best adapter model...')
			pickle.dump(
				best_adapter,
				open(f"{adapter_save_path}/model.pkl","wb")
			)
			
			with open(f"{adapter_save_path}/C.txt", "w") as f:
				f.write(str(best_c))
				f.close()

			print('generating test set predictions...')
			pred_df = pd.DataFrame({
					'pred_probs':best_adapter.predict_proba(X_test)[:,1],
					'labels':y_test,
					'prediction_id':prediction_id_test,
					'task':task,
					'train_type':train_type,
					'pretrain_cohort':cohort,
					'phase':'test'
				})
			print('evaluating predictions...')
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
			
			print('saving predictions and evaluations...')
			pred_df.reset_index(drop=True).to_csv(f"{result_save_path}/preds.csv")
			df_test.reset_index(drop=True).to_csv(f"{result_save_path}/test_eval.csv")
