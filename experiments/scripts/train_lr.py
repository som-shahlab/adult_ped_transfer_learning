import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import argparse
import pickle
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import log_loss

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

parser.add_argument(
	'--verbose',
	type=int,
	default=0
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
	
	cohort = cohort.merge(train_rows, how='left', on='prediction_id')
	cohort['train_row_idx'] = cohort['train_row_idx'].fillna(-1).astype(int)
	
	val_feats = sp.load_npz(f'{fn}/train/{args.feat_group}_feats.npz')
	val_rows = pd.read_csv(f'{fn}/val/val_pred_id_map.csv')
	
	cohort = cohort.merge(val_rows, how='left', on='prediction_id')
	cohort['val_row_idx'] = cohort['val_row_idx'].fillna(-1).astype(int)
	
	return train_feats.todense(), val_feats.todense(), cohort

def get_model(args, hp):
	# Create LR model using SGDClassifier so that partial_fit() can be called later for transfer 
	# learning purposes.
	return SGDClassifier(
					loss = hp['loss'], 
					random_state = hp['random_state'], 
					alpha = hp['C'], 
					penalty = hp['penalty'], 
					max_iter = hp['max_iter'],
					verbose = args.verbose
				)

def train_model(args, hp):
	print('Initialized model with hyperparams:')
	print(hp)
	model_save_path = f'{args.model_path}/{task}/lr_{hp["penalty"]}_{hp["C"]}_{args.feat_group}'
	os.makedirs(model_save_path,exist_ok=True)

	model = get_model(args, hp)

	print('Training...')
	model.fit(train_X, list(train_labels[task]))

	with open(f'{model_save_path}/model.pkl', 'wb') as pkl_file:
		pickle.dump(model, pkl_file)

	print('Evaluating...')
	val_preds = model.predict_proba(val_X)
	val_preds = [r[1] for r in val_preds]

	loss = log_loss(list(val_labels[f'{task}']),val_preds)
	print(f'Validation loss: {loss}')

	val_df = pd.DataFrame({'val_preds':val_preds, 'labels':list(val_labels[f'{task}']), 'prediction_id':list(val_labels['prediction_id'])})
	val_df.to_csv(f'{model_save_path}/val_preds.csv',index=False)

	return model, loss, val_df

def get_labels(args, task, cohort):
	#MAYBE ADD TEST EVALUATION IN THIS SCRIPT AS WELL
	train_labels = cohort.query(f'train_row_idx>=0 and {task}_fold_id!="ignore"')[['prediction_id', 'train_row_idx', f'{task}']]
	val_labels = cohort.query(f'val_row_idx>=0 and {task}_fold_id!="ignore"')[['prediction_id', 'val_row_idx', f'{task}']]
	return train_labels, val_labels

def slice_sparse_matrix(mat, rows):
	'''
	Slice rows in sparse matrix using given rows indices
	'''
	mask = np.zeros(mat.shape[0], dtype=bool)
	mask[rows] = True
	w = np.flatnonzero(mask)
	sliced = mat[w,:]
	return sliced

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
		print(f'Training models for {task} task...')
		train_labels, val_labels = get_labels(args, task, cohort)
		train_X = slice_sparse_matrix(train_data, list(train_labels['train_row_idx']))
		val_X = slice_sparse_matrix(val_data, list(val_labels['val_row_idx']))
		best_save_path = f'{args.model_path}/{task}/best/{args.feat_group}'
		os.makedirs(best_save_path, exist_ok=True)
		best_model = None
		best_loss = 999999999`
		best_val_preds = None
		best_hp = None
		
		for i, hp in enumerate(grid):
			model, loss, val_preds = train_model(args, hp)
			if loss < best_loss:
				print(f'Saving model as best {task} model...')
				best_model = model
				best_loss = loss
				best_val_preds = val_preds
				best_hp = hp
				
		with open(f'{best_save_path}/model.pkl', 'wb') as pkl_file:
			pickle.dump(best_model, pkl_file)
		
		val_preds.to_csv(f'{best_save_path}/val_preds.csv',index=False)
		
		with open(f'{best_save_path}/hp.yml','wb') as file:
			yaml.dump(best_hp, file)