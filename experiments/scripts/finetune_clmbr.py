import os
import json
import argparse
import shutil
import yaml
import random
import copy
from datetime import datetime

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import Trainer
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
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
    default=512,
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
    default=0.001,
    help='Learning rate for pretrained model.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
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

class MLPLayer(nn.Module):
	"""
	Linear classifier layer.
	"""
	def __init__(self, size):
		super().__init__()
		self.dense = nn.Linear(size, size)
		self.activation = nn.Tanh()
		
	def forward(self, features):
		x = self.dense(features)
		x = self.activation(x)
		
		return x

class EarlyStopping():
	def __init__(self, patience):
		self.patience = patience
		self.early_stop = False
		self.best_loss = 9999999
		self.counter = 0
	def __call__(self, loss):
		if self.best_loss > loss:
			self.counter = 0
			self.best_loss = loss
		else:
			self.counter += 1
			if self.counter == self.patience:
				self.early_stop = True
		return self.early_stop

def load_data(args, task):
	"""
	Load datasets from split csv files.
	"""

	data_path = f'{args.labelled_fpath}/{task}/ped'

	
	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')

	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())

	return train_data, val_data
        
def finetune_model(args, model, dataset, clmbr_save_path, clmbr_model_path):
	"""
	Finetune CLMBR model using linear layer.
	"""
	model.train()
	config = model.clmbr_model.config
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=lr)
	early_stop = EarlyStopping(args.patience)
	best_val_loss = 9999999
	best_epoch = 0
	
	for e in range(args.epochs):
		
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		
		model.train()
		pat_info_df = pd.DataFrame()
		model_train_loss_df = pd.DataFrame()
		model_val_loss_df = pd.DataFrame()
		train_loss = []
		train_preds = []
		train_lbls = []
		with DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in tqdm(train_loader):
				optimizer.zero_grad()
				outputs = model(batch)

				train_preds.extend(list(outputs['preds'].detach().clone().cpu().numpy()))
				train_lbls.extend(list(outputs['labels'].detach().clone().cpu().numpy()))
				loss = outputs["loss"]

				loss.backward()
				optimizer.step()
				train_loss.append(loss.item())
		print('Training loss:',  np.sum(train_loss))

		# evaluate on validation set
		val_preds, val_lbls, val_losses = evaluate_model(args, model, dataset, e)
		print('Validation loss:',  np.sum(val_losses))

		# Save train and val model predictions/labels
		df = pd.DataFrame({'epoch':e,'preds':train_preds,'labels':train_lbls})
		df.to_csv(f'{clmbr_save_path}/{e}/train_preds.csv', index=False)
		df = pd.DataFrame({'epoch':e,'preds':val_preds,'labels':val_lbls})
		df.to_csv(f'{clmbr_save_path}/{e}/val_preds.csv', index=False)
		
		#save current epoch model
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		torch.save(model.clmbr_model.state_dict(), os.path.join(clmbr_save_path,f'{e}/best'))
		shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/{e}/info.json')
		with open(f'{clmbr_save_path}/{e}/config.json', 'w') as f:
			json.dump(config,f)			
		
		#save model as best model if condition met
		if scaled_val_loss < best_val_loss:
			best_val_loss = scaled_val_loss
			best_epoch = e
			best_model = copy.deepcopy(model.clmbr_model)
		
		# Trigger early stopping if model hasn't improved for awhile
		if early_stop(scaled_val_loss):
			print(f'Early stopping at epoch {e}')
			break
	
	pd.DataFrame({'loss':train_loss}).to_csv(f'{clmbr_save_path}/train_loss.csv', index=True)
	pd.DataFrame({'loss':val_loss}).to_csv(f'{clmbr_save_path}/val_loss.csv', index=True)
	
	# save best epoch for debugging 
	with open(f'{clmbr_save_path}/best_epoch.txt', 'w') as f:
		f.write(f'{best_epoch}')
		
	return best_model, best_val_loss, best_epoch

def evaluate_model(args, model, dataset, e):
	model.eval()
	
	criterion = nn.CrossEntropyLoss()
	
	preds = []
	lbls = []
	losses = []
	pat_info_df = pd.DataFrame()
	with torch.no_grad():
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device) as eval_loader:
			for batch in tqdm(eval_loader):
				outputs = model(batch)
				loss = outputs["loss"]
				losses.append(loss.item())
				preds.extend(list(outputs['preds'].cpu().numpy()))
				lbls.extend(list(outputs['labels'].cpu().numpy()))

	
	return preds, lbls, losses

def finetune(args, cl_hp, clmbr_model_path, dataset):
	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)
	# Modify CLMBR config settings
	clmbr_model.config["model_dir"] = clmbr_save_path
	clmbr_model.config["batch_size"] = cl_hp['batch_size']
	clmbr_model.config["epochs_per_cycle"] = args.epochs
	clmbr_model.config["warmup_epochs"] = 1

	config = clmbr_model.config

	clmbr_model.unfreeze()
	# Get contrastive learning model 
	clmbr_model.train()

	# Run finetune procedure
	clmbr_model, val_loss, best_epoch = finetune_model(args, clmbr_model, dataset, clmbr_save_path, clmbr_model_path)
	clmbr_model.freeze()

	# Save model and save info and config to new model directory for downstream evaluation
	torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,'best'))
	shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/info.json')
	with open(f'{clmbr_save_path}/config.json', 'w') as f:
		json.dump(config,f)
	
	return clmbr_model, val_loss, best_epoch

def calc_metrics(args, df):
	evaluator = StandardEvaluator()
	eval_ci_df, eval_df = evaluator.bootstrap_evaluate(
		df,
		n_boot = args.n_boot,
		n_jobs = args.n_jobs,
		strata_vars_eval = ['phase'],
		strata_vars_boot = ['labels'],
		patient_id_var='person_id',
		return_result_df = True
	)
	eval_ci_df['model'] = 'probe'
	return eval_ci_df

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	
	clmbr_grid = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.hparams_fpath,'gru')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)

	clmbr_model_path = f'{args.pt_model_path}/gru_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
	print(clmbr_model_path)
	best_val_loss = 9999999
	best_params = None
	
	train_data, val_data = load_data(args, clmbr_hp)

	dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
									 args.extract_path + '/ontology.db', 
									 f'{clmbr_model_path}/info.json', 
									 train_data, 
									 val_data )
	
	for j, hp in enumerate(clmbr_grid):
		print('finetuning model with params: ', cl_hp)
		best_ft_path = f"{args.model_path}/gru_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/best"
		clmbr_save_path = f"{args.model_path}/gru_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}"
		print(clmbr_save_path)
	
		os.makedirs(f"{clmbr_save_path}",exist_ok=True)
	
		
		os.makedirs(f"{best_ft_path}",exist_ok=True)
		
		model, val_loss, best_epoch = finetune(args, cl_hp, clmbr_model_path, dataset)
		
		if val_loss < best_val_loss:
			print('Saving as best finetuned model...')
			best_val_loss = val_loss
			best_params = cl_hp

			torch.save(model.state_dict(), os.path.join(best_ft_path,'best'))
			shutil.copyfile(f'{clmbr_model_path}/info.json', f'{best_ft_path}/info.json')
			with open(f'{best_ft_path}/config.json', 'w') as f:
				json.dump(model.config,f)
			with open(f"{best_ft_path}/hyperparams.yml", 'w') as file: 
				yaml.dump(best_params,file)
			with open(f'{best_ft_path}/best_epoch.txt', 'w') as f:
				f.write(f'{best_epoch}')
        
    