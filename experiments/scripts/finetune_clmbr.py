import os
import json
import argparse
import shutil
import yaml
import random
import copy
from datetime import datetime
import logging

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import Trainer
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader
from ehr_ml.clmbr.utils import read_config, read_info
from ehr_ml.utils import set_up_logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
from prediction_utils.util import str2bool
#from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '--pt_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr/pretrained/models',
    help='Base path for the pretrained model.'
)

parser.add_argument(
    '--ft_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr/finetuned/models',
    help='Base path for the best finetuned model.'
)

parser.add_argument(
    '--info_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr/pretrained/info/finetune/info.json',
    help='Base path for the pretrained model.'
)

parser.add_argument(
    '--extract_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/extracts/20220801',
    help='Base path for the extracted database.'
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort",
    help='Base path for cohort file'
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/hyperparams",
    help='Base path for hyperparameter files'
)

parser.add_argument(
    '--labelled_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/clmbr_labelled_data",
    help='Base path for labelled data directory'
)

parser.add_argument(
    '--train_end_date',
    type=str,
    default='2020-12-31',
    help='End date of training ids.'
)

parser.add_argument(
    '--val_end_date',
    type=str,
    default='2020-12-31',
    help='End date of validation ids.'
)

parser.add_argument(
    '--cohort_dtype',
    type=str,
    default='parquet',
    help='Data type for cohort file.'
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=4000,
    help='Size of training batch.'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=2000,
    help='Size of training batch.'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of training epochs.'
)

parser.add_argument(
    '--size',
    type=int,
    default=800,
    help='Size of embedding vector.'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.000001,
    help='Learning rate for pretrained model.'
)

parser.add_argument(
    '--dropout',
    type=str,
    default="0",
    help='Dropout for pretrained model.'
)

parser.add_argument(
    '--l2',
    type=str,
    default="0",
    help='L2 regularization parameter.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
    '--cohort_type',
    type=str,
    default='all',
    help='Cohort CLMBR model was pretrained on'
)

parser.add_argument(
	'--patience',
	type=int,
	default=5,
	help='Number of epochs to wait before triggering early stopping.'
)

parser.add_argument(
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)

parser.add_argument(
    '--overwrite',
    type=str2bool,
    default='true'
)

parser.add_argument(
    '--logging',
    type=str2bool,
    default="True",
    help='Whether to run logging'
)
        
def finetune(args, cl_hp, clmbr_save_path):
	# Run finetuning on pretrained CLMBR model
	og_model_path = f"{clmbr_save_path}/original_model"
	ft_model_path = f"{clmbr_save_path}/finetune_model"
	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(og_model_path, args.device)

	# Modify CLMBR config settings
	clmbr_model.config["model_dir"] = ft_model_path
	clmbr_model.config["batch_size"] = cl_hp['batch_size']
	clmbr_model.config["eval_batch_size"] = cl_hp['eval_batch_size']
	clmbr_model.config["epochs_per_cycle"] = args.epochs
	clmbr_model.config['early_stopping_patience'] = args.patience

	trainer = Trainer(clmbr_model)
	
	# set up logging if flag set
	if args.logging:
		set_up_logging(os.path.join(ft_model_path, "finetune.log"))
		logging.info("Args: %s", str(clmbr_model.config))
		
	# overwrite ft model save directory if flag set
	# if directory is populated and overwrite == False
	# CLMBR model will throw error when saving after first epoch
	if args.overwrite and os.path.exists(ft_model_path):
		shutil.rmtree(ft_model_path, ignore_errors=True)
		os.makedirs(ft_model_path, exist_ok=True)
	elif not os.path.exists(ft_model_path):
		os.makedirs(ft_model_path, exist_ok=True)
	
	dataset = PatientTimelineDataset(
		os.path.join(args.extract_path, "extract.db"),
		os.path.join(args.extract_path, "ontology.db"),
		os.path.join(og_model_path, "info.json"),
	)
	
	trainer.train(dataset, use_pbar=False)

def prep_finetune_dir(args, source_path, target_path):
	# Copy pretrained model directory to finetuned directory subfolder and
	# replace info.json with the finetune-specific info.json
	t_path = f"{target_path}/original_model"
	os.makedirs(t_path,exist_ok=True)
	shutil.copytree(source_path, t_path, dirs_exist_ok=True)
	shutil.copyfile(args.info_path, f'{t_path}/info.json')

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	
	cohort_type = args.cohort_type
	
	clmbr_hp = {'size':args.size,'epochs':args.epochs, 'batch_size':args.batch_size, 'eval_batch_size':args.eval_batch_size, 'lr':args.lr, 'dropout':args.dropout, 'l2':args.l2 }
	print('finetuning model with params: ', clmbr_hp)
	clmbr_model_path = f'{args.pt_model_path}/{cohort_type}/gru_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'

	clmbr_save_path = f"{args.ft_model_path}/{cohort_type}/gru_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}"
	print(f"saving to {clmbr_save_path}")

	prep_finetune_dir(args,clmbr_model_path, clmbr_save_path)
	
	print("finetuning model...")
	finetune(args, clmbr_hp, clmbr_save_path)
    