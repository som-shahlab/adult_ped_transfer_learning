import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import joblib
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
	'--sparse_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/merged_features/features_sparse'
)
parser.add_argument(
	'--vocab_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/merged_features/vocab'
)
parser.add_argument(
	'--cohort_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort'
)
parser.add_argument(
	'--bin_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bin_features'
)
parser.add_argument(
	'--save_full',
	action='store_true'
)

parser.add_argument(
    '--train_start_year',
    type=int,
    default=2008,
    help='Start date of training ids.'
)

parser.add_argument(
    '--train_end_year',
    type=int,
    default=2019,
    help='End date of training ids.'
)

parser.add_argument(
    '--val_start_year',
    type=int,
    default=2020,
    help='Start date of validation ids.'
)

parser.add_argument(
    '--val_end_year',
    type=int,
    default=2020,
    help='End date of validation ids.'
)

parser.add_argument(
    '--test_start_year',
    type=int,
    default=2021,
    help='Start date of test ids.'
)

parser.add_argument(
    '--test_end_year',
    type=int,
    default=2022,
    help='End date of test ids.'
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
	
def split_bin_array(args, bin_arr, vocab, cohort_df, age_group='pediatric'):
	'''
	Split full age group stratified feature set into train/val/test splits.
	Generate splits that consist of shared features only, all adult features only and all pediatric features only.
	Also save new vocabulary for pruned feature sets.
	'''
	print(f'Splitting {age_group} features...\n')
	test_df = cohort_df.query('fold_id=="test" and admission_year>=@args.test_start_year and admission_year<=@args.test_end_year')
	val_df = cohort_df.query('fold_id=="val" and admission_year>=@args.val_start_year and admission_year<=@args.val_end_year')
	train_df = cohort_df.query('fold_id!="test" and fold_id!="val" and admission_year>=@args.train_start_year and admission_year<=@args.train_end_year')
	
	shared_feat_cols = pd.read_csv(f'{args.bin_path}/shared_feats.csv')
	pediatric_feat_cols = pd.read_csv(f'{args.bin_path}/only_pediatric_feats.csv')
	adult_feat_cols = pd.read_csv(f'{args.bin_path}/only_adult_feats.csv')
	
	all_ped_cols = list(shared_feat_cols['feat_indices']) + list(pediatric_feat_cols['feat_indices'])
	all_ped_cols.sort()
	all_ad_cols = list(shared_feat_cols['feat_indices']) + list(adult_feat_cols['feat_indices'])
	all_ad_cols.sort()
	shared_cols = list(shared_feat_cols['feat_indices'])
	shared_cols.sort()
	
	for split in ['train','val','test']:
		os.makedirs(f'{args.bin_path}/{age_group}/{split}', exist_ok=True)
		if split == 'train':
			pt_rows = list(train_df['features_row_id'])
			print(f'# train samples: {len(pt_rows)}')
			row_map = pd.DataFrame({f'{split}_row_idx':[i for i in range(len(pt_rows))], 'prediction_id':list(train_df['prediction_id'])})
			row_map.to_csv(f'{args.bin_path}/{age_group}/{split}/train_pred_id_map.csv',index=False)
		elif split == 'val':
			pt_rows = list(val_df['features_row_id'])
			print(f'# val samples: {len(pt_rows)}')
			row_map = pd.DataFrame({f'{split}_row_idx':[i for i in range(len(pt_rows))], 'prediction_id':list(val_df['prediction_id'])})
			row_map.to_csv(f'{args.bin_path}/{age_group}/{split}/val_pred_id_map.csv',index=False)
		elif split == 'test':
			pt_rows = list(test_df['features_row_id'])
			print(f'# test samples: {len(pt_rows)}')
			row_map = pd.DataFrame({f'{split}_row_idx':[i for i in range(len(pt_rows))], 'prediction_id':list(test_df['prediction_id'])})
			row_map.to_csv(f'{args.bin_path}/{age_group}/{split}/test_pred_id_map.csv',index=False)
		
		pt_feats = bin_arr[pt_rows]
		sp.save_npz(f'{args.bin_path}/{age_group}/{split}/all_feats.npz', pt_feats)

		for feat_type in ['shared', 'pediatric', 'adult']:
			print(f'Creating {feat_type} pruned feature sets...')
			col_list = shared_cols if feat_type == 'shared' else all_ped_cols if feat_type == 'pediatric' else all_ad_cols
			print('Pruning..')
			prune_feats, prune_vocab = prune_cols(pt_feats, shared_cols, vocab)
			print('Saving features...')
			sp.save_npz(f'{args.bin_path}/{age_group}/{split}/{feat_type}_feats.npz', prune_feats)
			print('Saving vocab...\n')
			prune_vocab.to_parquet(f'{args.bin_path}/{age_group}/{split}/{feat_type}_feats_vocab.parquet', engine='pyarrow', index=False)
			
def prune_cols(feats, col_list, vocab):
	'''
	Select only feature columns in col_list.
	'''
	pruned_feats = feats.tocsc()[:,col_list]
	pruned_vocab = vocab[vocab['col_id'].isin(col_list)]['feature_id'].reset_index(drop=True).reset_index().rename(columns={'index':'col_id'})
	return pruned_feats.tocsr(), pruned_vocab
					
if __name__=='__main__':
	args = parser.parse_args()	
	feats_id_map = read_file(
			os.path.join(
				args.sparse_path,
				"features_row_id_map.parquet"
			),
			engine='pyarrow'
		)

	vocab = read_file(
			os.path.join(
				args.vocab_path,
				"vocab.parquet"
			),
			engine='pyarrow'
	)
	cohort = read_file(
			os.path.join(
				args.cohort_path,
				"cohort_split_no_nb.parquet"
			),
			engine='pyarrow'
		)
	cohort = cohort.merge(feats_id_map, how='left', on='prediction_id')
	features = joblib.load(os.path.join(args.sparse_path,"features.gz"))
	ped_df = cohort[cohort['adult_at_admission'] == 0].sort_values(by='features_row_id')
	adult_df = cohort[cohort['adult_at_admission'] == 1].sort_values(by='features_row_id')
	split_bin_array(args, features, vocab, ped_df, 'pediatric')
	split_bin_array(args, features, vocab, adult_df, 'adult')
	split_bin_array(args, features, vocab, cohort, 'shared')

	if args.save_full: 
		# Save full age group stratified feature sets without train/val/test split
		print('Saving full, unsplit feature sets...')
		ped_rows = list(ped_df['features_row_id'])
		ad_rows = list(adult_df['features_row_id'])

		ped_row_map = {'row_idx_og':ped_rows,'row_idx_new':[i for i in range(len(ped_rows))]}
		ad_row_map = pd.DataFrame({'row_idx_og':ad_rows,'row_idx_new':[i for i in range(len(ad_rows))]})
		
		print('Saving pediatric features...')
		pd.DataFrame(ped_row_map).to_csv(args.bin_path + '/pediatric/pat_map.csv',index=False)		
		sp.save_npz(f'{args.bin_path}/pediatric/full.npz', features[ped_rows])
		print('Saving adult features...')
		pd.DataFrame(ad_row_map).to_csv(args.bin_path + '/adult/pat_map.csv',index=False)
		sp.save_npz(f'{args.bin_path}/adult/full.npz', features[ad_rows])

