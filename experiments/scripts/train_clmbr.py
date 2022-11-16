import os
import argparse
import yaml
import shutil
import pdb
import torch
import joblib

import pandas as pd
import numpy as np

from itertools import zip_longest
from subprocess import (run, Popen)
from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    description='Pretrain baseline CLMBR model'
)

parser.add_argument(
    '--min_patient_count', 
    type=str,
    default="100",
)

parser.add_argument(
    '--extracts_fpath', 
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/extracts/20220801",
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort",
)

parser.add_argument(
    '--infos_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr/pretrained/info"
)

parser.add_argument(
    '--models_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr/pretrained/models"
)

parser.add_argument(
    '--train_start_date',
    type=str,
    default='2008-01-01',
    help='Start date of training ids.'
)

parser.add_argument(
    '--train_end_date',
    type=str,
    default='2020-12-31',
    help='End date of training ids.'
)

parser.add_argument(
    '--val_start_date',
    type=str,
    default='2008-01-01',
    help='Start date of validation ids.'
)

parser.add_argument(
    '--val_end_date',
    type=str,
    default='2020-12-31',
    help='End date of validation ids.'
)

parser.add_argument(
    '--test_start_date',
    type=str,
    default='2021-01-01',
    help='Start date of test ids.'
)

parser.add_argument(
    '--test_end_date',
    type=str,
    default='2022-08-01',
    help='End date of test ids.'
)

parser.add_argument(
    '--excluded_patient_path',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/held_out_patients/excluded_patient_ids"
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/hyperparams/"
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Encoder type: GRU/Transformer',
)

parser.add_argument(
    '--overwrite',
    type=str2bool,
    default='true'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=1
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=8
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=7
)

parser.add_argument(
	'--seed',
	type=int,
	default=44
)

parser.add_argument(
	'--device',
	type=str,
	default='cuda:0'
)

parser.add_argument(
	'--pretrain_group',
	type=str,
	default='mix'
)

parser.add_argument(
	'--early_stopping',
	action='store_true'
)

if __name__ == "__main__":
    
	args = parser.parse_args()
	
	# threads
	torch.set_num_threads(1)
	joblib.Parallel(n_jobs=1)

	grid = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.hparams_fpath,args.encoder)}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)
    
	# create info
	info_dir=f'{args.infos_fpath}/{args.pretrain_group}'
	train_end_date=args.train_end_date
	val_end_date=args.val_end_date
	
	print(args.pretrain_group)
	exc_pat_file = f"{args.excluded_patient_path}_{args.pretrain_group}.txt"
	print(exc_pat_file)
	if args.overwrite and os.path.exists(info_dir):
		shutil.rmtree(info_dir, ignore_errors=True)

	run([
		'clmbr_create_info',
		f"{args.extracts_fpath}",
		f"{info_dir}",
		f"{train_end_date}",
		f"{val_end_date}",
		"--train_start_date", f"{args.train_start_date}",
		"--val_start_date", f"{args.val_start_date}",
		"--min_patient_count", args.min_patient_count,
		"--excluded_patient_file", exc_pat_file,
		"--seed", f'{args.seed}'
	])
	
	processes=[]
    
    # collect args
	for i,hparams in enumerate(grid):
        
		model_dir=f'{args.models_fpath}/{args.pretrain_group}/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}'

		if args.overwrite and os.path.exists(model_dir):
			shutil.rmtree(model_dir, ignore_errors=True)
			os.makedirs(model_dir, exist_ok=True)
        
		torch.manual_seed(args.seed)
		
		p_args = [
			'clmbr_train_model',
			model_dir,
			info_dir,
			'--lr', f"{hparams['lr']}",
			'--encoder_type', f"{hparams['encoder_type']}",
			'--size', f"{hparams['size']}",
			'--dropout', f"{hparams['dropout']}",
			'--batch_size', f"{hparams['batch_size']}",
			'--epochs', f"{hparams['epochs']}",
			'--l2', f"{hparams['l2']}",
			'--warmup_epochs', f"{hparams['warmup_epochs']}",
			'--device', f'{args.device}',
			'--early_stopping_patience', f"{hparams['early_stopping_patience']}",
			'--early_stopping' if args.early_stopping==True else ''
		]
        
		processes.append(p_args)

    # group processes 
	processes = [
		(
			Popen(
				p,
				env=dict(os.environ, CUDA_VISIBLE_DEVICES = str(i%args.n_gpu+args.gpu_num_start))
			) for i,p in enumerate(processes)
		)
	] * args.n_jobs

    # submit n_jobs jobs at a time
	for sub_p in zip_longest(*processes): 
		for p in filter(None, sub_p):
			p.wait()