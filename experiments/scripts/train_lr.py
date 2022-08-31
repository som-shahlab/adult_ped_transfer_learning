import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import argparse
import pickle
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid

parser=argparse.ArgumentParser()

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
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/baseline/lr'
)

parser.add_argument(
	'--results_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results/baseline/lr'
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

def load_data(args):
	
	cohort = read_file(
			os.path.join(
				args.cohort_path,
				"cohort_split.parquet"
			),
			engine='pyarrow'
		)
	
	fn = f'{args.bin_path}/{args.cohort_type}'
	
	train_feats = sp.load_npz(f'{fn}/train/{args.feat_group}_feats.npz')
	
	train_rows = pd.read_csv(f'{fn}/train/train_pred_id_map.csv')
	print(cohort.columns)
	print(train_rows.head())
	
	cohort = cohort.merge(train_rows, how='left', on='prediction_id')
	print(cohort)
	print(Sadasd)
	val_feats = sp.load_npz(f'{fn}/train/{args.feat_group}_feats.npz')
	val_rows = pd.read_csv(f'{fn}/val/val_pred_id_map.csv')
	
	cohort = cohort.merge(val_rows, how='left', on='prediction_id')
	
	return train, val, cohort

def get_model(args, hp):
	# Create LR model using SGDClassifier so that partial_fit() can be called later for transfer 
	# learning purposes.
	return SGDClassifier(
					loss = hp['loss'], 
					 random_state = hp['random_state'], 
					 alpha = hp['C'], 
					 penalty=hp['penalty'], 
					 max_iter = hp['max_iter']
				)
def get_labels(args, task, cohort):
	train_labels = cohort[['prediction_id', f'{task}']]

if __name__ == '__main__':
	args = parser.parse_args()
	
	grid = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.hparam_path,'lr')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)
	
	train_data, val_data, cohort = load_data(args)
	
	for task in args.tasks:
		train_labels, val_labels = get_labels(args, task, cohort)
		best_save_path = f'{args.model_path}/{args.task}/best'
		os.makedirs(best_save_path, exist_ok=True)
		best_model = None
		best_loss = 999999999
		best_loss_curve = None
		best_val_preds = None
		for i, hp in enumerate(grid):
			model_save_path = f'{args.model_path}/{args.task}/lr_{hp["penalty"]}_{hp["C"]}'
			os.makedirs(model_save_path,exist_ok=True)
			
			model = get_model(args, hp)
			
			model.fit(train_data, train_labels)
			losses = model.loss_curve_
			pkl_file = open(f'{model_save_path}/model.pkl', 'wb')
			pickle.dump(pkl_file)
			pkl_file.close()
			
			val_preds = model.predict_proba(val_data)
			
			val_df = pd.DataFrame({'val_preds':val_preds, 'labels':val_labels, 'prediction_id':val_pred_ids})
			
			val_df.to_csv(f'{model_save_path}/val_preds.csv',index=False)
			
			if losses[-1] < best_loss:
				best_model = model
				best_loss = losses[-1]
				best_loss_curve = losses
				best_val_preds = val_df
		
		pkl_file = open(f'{best_save_path}/model.pkl', 'wb')
		pickle.dump(pkl_file)
		pkl_file.close()
		
		val_df.to_csv(f'{best_save_path}/val_preds.csv',index=False)
		pd.DataFrame({'loss':best_loss_curve}).to_csv(f'{best_save_path}/loss.csv')