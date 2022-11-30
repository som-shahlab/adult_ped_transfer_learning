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
    default=128,
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

class LinearAdapter(nn.Module):
	def __init__(self, clmbr_model, size, device='cuda:0'):
		super().__init__()
		self.clmbr_model = clmbr_model
		self.config = clmbr_model.config

		self.dense = nn.Linear(size,1)
		self.activation = nn.Sigmoid()

		self.device = torch.device(device)

	def forward(self, batch):
		features = self.clmbr_model.timeline_model(batch['rnn']).to(self.device)

		label_indices, label_values = batch['label']

		flat_features = features.view((-1, features.shape[-1]))
		target_features = F.embedding(label_indices, flat_features).to(self.device)

		preds = self.activation(self.dense(target_features))
		return preds, label_values.to(torch.float32)

	def freeze_clmbr(self):
		self.clmbr_model.freeze()

	def unfreeze_clmbr(self):
		self.clmbr_model.unfreeze()

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

def load_data(args, task='hospital_mortality'):
	"""
	Load datasets from split csv files.
	For inital finetunign any task works as the labels are ignored
	"""

	data_path = f'{args.labelled_fpath}/{task}/ped'

	
	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')
	test_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_test.csv')

	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')
	test_days = pd.read_csv(f'{data_path}/day_indices_test.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')
	test_labels = pd.read_csv(f'{data_path}/labels_test.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())
	test_data = (test_labels.to_numpy().flatten(),test_pids.to_numpy().flatten(),test_days.to_numpy().flatten())

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())
	test_data = (test_labels.to_numpy().flatten(),test_pids.to_numpy().flatten(),test_days.to_numpy().flatten())

	return train_data, val_data, test_data
        
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
		
		epoch_train_loss = 0.0
		epoch_val_loss = 0.0
		
		with DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in tqdm(train_loader):
				optimizer.zero_grad()
				outputs = model(batch)

				loss = outputs["loss"]

				loss.backward()
				optimizer.step()
				epoch_train_loss += loss.item()
		print('Training loss:',  epoch_train_loss)
		
		with torch.no_grad():
			with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config["batch_size"], device=args.device) as val_loader:
				for batch in val_loader:
					outputs = model(batch)
					loss = outputs["loss"]
					
					epoch_val_loss += loss.item()

		print('Validation loss:',  epoch_val_loss)
		
		#save current epoch model
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		torch.save(model.clmbr_model.state_dict(), os.path.join(clmbr_save_path,f'{e}/best'))
		shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/{e}/info.json')
		with open(f'{clmbr_save_path}/{e}/config.json', 'w') as f:
			json.dump(config,f)			
		
		#save model as best model if condition met
		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			best_epoch = e
			best_model = copy.deepcopy(model.clmbr_model)
		
		# Trigger early stopping if model hasn't improved for awhile
		if early_stop(scaled_val_loss):
			print(f'Early stopping at epoch {e}')
			break
	
	# save best epoch for debugging 
	with open(f'{clmbr_save_path}/best_epoch.txt', 'w') as f:
		f.write(f'{best_epoch}')
		
	return best_model, best_val_loss, best_epoch

def evaluate_adapter(args, model, dataset):
	model.eval()

	criterion = nn.BCELoss()

	preds = []
	lbls = []
	ids = []
	with torch.no_grad():
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device) as eval_loader:
			for batch in eval_loader:
				logits, labels = model(batch)
				loss = criterion(logits, labels.unsqueeze(-1))
				# losses.append(loss.item())
				preds.extend(logits.cpu().numpy().flatten())
				lbls.extend(labels.cpu().numpy().flatten())
				ids.extend(batch['pid'])
	return preds, lbls, ids

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
	
	tasks = ['hospital_mortality','sepsis','LOS_7','readmission_30','icu_admission','aki1_label','aki2_label','hg_label','np_500_label','np_1000_label']
	
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

	
	train_data, val_data, test_data = load_ft_data(args)
	
	for cohort_type in ['ad', 'all']:
	
		for j, hp in enumerate(clmbr_grid):
			clmbr_model_path = f'{args.pt_model_path}/{cohort_type}/gru_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
			print(clmbr_model_path)

			train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 train_data, 
										 val_data )

			test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 train_data, 
										 test_data )

			print('finetuning model with params: ', hp)
			best_ft_path = f"{args.ft_model_path}/{cohort_type}/gru_sz_{hp['size']}_do_{hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/best"
			clmbr_save_path = f"{args.ft_model_path}/{cohort_type}/gru_sz_{hp['size']}_do_{hp['dropout']}_lr_{hp['lr']}_l2_{hp['l2']}"
			print(f"saving to {clmbr_save_path}")

			os.makedirs(f"{clmbr_save_path}",exist_ok=True)

			os.makedirs(f"{best_ft_path}",exist_ok=True)

			print("finetuning model")
			model, val_loss, best_epoch = finetune(args, cl_hp, clmbr_model_path, train_dataset)

			print('Saving finetuned model...')
			torch.save(model.state_dict(), os.path.join(best_ft_path,'best'))
			shutil.copyfile(f'{clmbr_model_path}/info.json', f'{best_ft_path}/info.json')
			with open(f'{best_ft_path}/config.json', 'w') as f:
				json.dump(model.config,f)
			with open(f"{best_ft_path}/hyperparams.yml", 'w') as file: 
				yaml.dump(hp,file)
			with open(f'{best_ft_path}/best_epoch.txt', 'w') as f:
				f.write(f'{best_epoch}')
			
			#eval finetuned model on 
			for task in tasks:
				print("evaluating finetuned model on task {task}")
				model.freeze()
				adapter_model = LinearAdapter(model, hp['size'])

				adapter_model.to(args.device)

				train_data, val_data, test_data = load_ft_data(args, task)
				
				train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 train_data, 
										 val_data )

				test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 train_data, 
										 test_data )

				print('training adapter...')
				# Train adapter and evaluate on validation 
				adapter_model, val_preds, val_labels, val_ids = train_adapter(args, adapter_model, train_dataset, adapter_save_path)

				val_df = pd.DataFrame({'CLMBR':'FT', 'model':'linear', 'task':task, 'cohort_type':cohort_type, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
				val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

				print('testing adapter...')
				test_preds, test_labels, test_ids = evaluate_adapter(args, adapter_model, test_dataset)

				test_df = pd.DataFrame({'CLMBR':'FT', 'model':'linear', 'task':task, 'cohort_type':cohort_type, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
				test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)
				df_preds = pd.concat((val_df,test_df))
				df_preds['CLMBR'] = df_preds['CLMBR'].astype(str)
				df_preds['model'] = df_preds['model'].astype(str)
				df_preds['task'] = df_preds['task'].astype(str)
				df_preds['cohort_type'] = cohort_type
				df_preds['phase'] = df_preds['phase'].astype(str)

				df_eval = calc_metrics(args, df_preds)
				df_eval['CLMBR'] = 'FT'
				df_eval['task'] = task
				df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)
    