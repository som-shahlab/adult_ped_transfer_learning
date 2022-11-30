import os
import json
import argparse
import shutil
import yaml
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
from sklearn.metrics import roc_auc_score
from prediction_utils.util import str2bool
#from torch.utils.data import DataLoader, Dataset
from prediction_utils.pytorch_utils.metrics import StandardEvaluator


parser = argparse.ArgumentParser()

parser.add_argument(
	'--pt_model_path',
	type=str,
	default='/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models/clmbr',
	help='Base path for the pretrained model.'
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
	'--cohort_dtype',
	type=str,
	default='parquet',
	help='Data type for cohort file.'
)

parser.add_argument(
	'--pretrain_group',
	type=str,
	default='mix',
	help='Pretrained cohort type.'
)

parser.add_argument(
	'--train_cohort',
	type=str,
	default='ped',
	help='Cohort to train adapter on.'
)

parser.add_argument(
	'--test_cohort',
	type=str,
	default='ped',
	help='Cohort to test adapter on.'
)

parser.add_argument(
	'--task',
	type=str,
	default='hospital_mortality',
	help='Task to evaluate.'
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
	'--n_boot',
	type=int,
	default=1000,
	help='Number of bootstrap iterations.'
)

parser.add_argument(
	'--n_jobs',
	type=int,
	default=8,
	help='Number of bootstrap jobs.'
)

parser.add_argument(
	'--early_stop',
	type=str2bool,
	default=True,
	help='Flag to enable early stopping.'
)

parser.add_argument(
	'--size',
	type=int,
	default=800,
	help='Size of embedding vector.'
)

parser.add_argument(
	'--dropout',
	type=float,
	default=0,
	help='Dropout proportion for training.'
)

parser.add_argument(
	'--lr',
	type=float,
	default=0.001,
	help='Learning rate for pretrained model.'
)

parser.add_argument(
	'--l2',
	type=float,
	default=0,
	help='L2 rate for pretrained model.'
)

parser.add_argument(
	'--encoder',
	type=str,
	default='gru',
	help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
	'--device',
	type=str,
	default='cuda:2',
	help='Device to run torch model on.'
)

parser.add_argument(
	'--patience',
	type=int,
	default=5,
	help='Number of epochs to wait before triggering early stopping.'
)

parser.add_argument(
	'--train_type',
	type=str,
	default='pretrained'
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
		self.best_loss = 99999999
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


def load_datasets(args, task, clmbr_model_path):
	"""
	Load datasets from split csv files.
	"""
	train_data_path = f'{args.labelled_fpath}/{task}/{args.train_cohort}'
	test_data_path = f'{args.labelled_fpath}/{task}/{args.test_cohort}'

	print(f'Loading data...')

	train_pids = pd.read_csv(f'{train_data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{train_data_path}/ehr_ml_patient_ids_val.csv')
	test_pids = pd.read_csv(f'{test_data_path}/ehr_ml_patient_ids_test.csv')

	train_days = pd.read_csv(f'{train_data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{train_data_path}/day_indices_val.csv')
	test_days = pd.read_csv(f'{test_data_path}/day_indices_test.csv')

	train_labels = pd.read_csv(f'{train_data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{train_data_path}/labels_val.csv')
	test_labels = pd.read_csv(f'{test_data_path}/labels_test.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())
	test_data = (test_labels.to_numpy().flatten(),test_pids.to_numpy().flatten(),test_days.to_numpy().flatten())

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
	print(f"{len(test_labels[test_labels[task]==1])} positive labels in test set")
	print(f"{len(test_labels[test_labels[task]==0])} negative labels in test set")
	return train_dataset, test_dataset


def train_adapter(args, model, dataset, save_path):
	"""
	Train linear classification adapter on frozen CLMBR model.
	At each epoch save model if validation loss was improved.
	"""
	model.train()
	model.freeze_clmbr()
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

	criterion = nn.BCELoss()

	early_stop = EarlyStopping(args.patience)

	best_model = None
	best_val_loss = 9999999
	best_val_preds = []
	best_val_lbls = []
	best_val_ids = []

	for e in range(args.epochs):
		val_preds = []
		val_lbls = []
		val_ids = []
		print(f'epoch {e+1}/{args.epochs}')
		epoch_train_loss = 0.0
		epoch_val_loss = 0.0
		# Iterate through training data loader
		with DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in train_loader:

				optimizer.zero_grad()
				logits, labels = model(batch)
				if torch.any(torch.isnan(logits)):
					return None, None, None, None, True
				loss = criterion(logits, labels.unsqueeze(-1))

				loss.backward()
				optimizer.step()
				epoch_train_loss += loss.item()

		# Iterate through validation data loader
		with torch.no_grad():
			with DataLoader(dataset, 9262, is_val=True, batch_size=model.config["batch_size"], device=args.device) as val_loader:
				for batch in val_loader:
					logits, labels = model(batch)
					loss = criterion(logits, labels.unsqueeze(-1))
					epoch_val_loss += loss.item()
					val_preds.extend(logits.cpu().numpy().flatten())
					val_lbls.extend(labels.cpu().numpy().flatten())
					val_ids.extend(batch['pid'])
				# val_losses.append(epoch_val_loss)

		#print epoch losses
		print('epoch train loss:', epoch_train_loss)
		print('epoch val loss:', epoch_val_loss)

		# save model if validation loss is improved
		if epoch_val_loss < best_val_loss:
			print('Saving best model...')
			best_val_loss = epoch_val_loss
			best_model = copy.deepcopy(model)
			torch.save(best_model, f'{save_path}/best_model.pth')

			# flatten prediction and label arrays
			best_val_preds = val_preds
			best_val_lbls = val_lbls
			best_val_ids = val_ids

		# Trigger early stopping if model hasn't improved for awhile
		if early_stop(epoch_val_loss):
			print(f'Early stopping at epoch {e}')
			break

	return best_model, best_val_preds, best_val_lbls, best_val_ids, False

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
				preds.extend(logits.cpu().numpy().flatten())
				lbls.extend(labels.cpu().numpy().flatten())
				ids.extend(batch['pid'])
	return preds, lbls, ids

def calc_metrics(args, df):
	evaluator = StandardEvaluator()
	df_test = evaluator.evaluate(
		df,
		strata_vars_eval=['phase'],
		label_var=['labels'],
		pred_prob_var=['pred_probs']
	)
	df_test['model'] = 'adapter'
	return df_test

if __name__ == '__main__':
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	hp = {"size":args.size, "dropout":args.dropout, "lr":args.lr, "l2":args.l2}

	# Iterate through tasks
	task = args.task
	print(f'Task {task}')
	cohort_type = args.pretrain_group
	print(f'Cohort type {cohort_type}')
	print(f'Training adapters with {args.train_cohort} dataset')

	# Path where CLMBR model is saved
	pt_model_str = f'gru_sz_{hp["size"]}_do_{hp["dropout"]}_lr_{hp["lr"]}_l2_{hp["l2"]}'
	clmbr_model_path = f'{args.pt_model_path}/{args.train_type}/models/{cohort_type}/{pt_model_str}'
	print(clmbr_model_path)

	# Load  datasets
	train_dataset, test_dataset = load_datasets(args, task, clmbr_model_path)

	# Path where CLMBR adapter will be saved
	adapter_save_path = f'{args.adapter_path}/{args.train_type}/adapters/{args.pretrain_group}/{task}/tr_{args.train_cohort}_tst_{args.test_cohort}/{pt_model_str}'
	os.makedirs(f"{adapter_save_path}",exist_ok=True)

	result_save_path = f'{args.results_path}/{args.train_type}/{args.pretrain_group}/{task}/tr_{args.train_cohort}_tst_{args.test_cohort}/{pt_model_str}'
	os.makedirs(f"{result_save_path}",exist_ok=True)
	print(adapter_save_path)
	print(result_save_path)

	# Load CLMBR model and attach linear adapter
	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
	clmbr_model.freeze()

	adapter_model = LinearAdapter(clmbr_model, hp['size'], args.device)

	adapter_model.to(args.device)

	print('Training adapter...')
	# Train adapter and evaluate on validation 
	adapter_model, val_preds, val_labels, val_ids, is_nan = train_adapter(args, adapter_model, train_dataset, adapter_save_path)

	if is_nan:
		print('Erroneous NaN output detected from model, skipping...')
		pass
	else:
		val_df = pd.DataFrame({'CLMBR':'PT', 'model':'linear', 'task':task, 'cohort_type':cohort_type, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
		val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

		print('Testing adapter...')
		test_preds, test_labels, test_ids = evaluate_adapter(args, adapter_model, test_dataset)

		test_df = pd.DataFrame({'CLMBR':'PT' if args.pretrain_group == 'pretrained' else 'FT', 'model':'linear', 'task':task, 'cohort_type':cohort_type, 'adapter_train': args.train_cohort, 'adapter_test':args.test_cohort, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
		test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)
		df_preds = pd.concat((val_df,test_df))
		df_preds['CLMBR'] = 'PT' if args.pretrain_group == 'pretrained' else 'FT'
		df_preds['model'] = df_preds['model'].astype(str)
		df_preds['task'] = df_preds['task'].astype(str)
		df_preds['cohort_type'] = cohort_type
		df_preds['adapter_train'] = args.train_cohort
		df_preds['adapter_test'] = args.test_cohort
		df_preds['phase'] = df_preds['phase'].astype(str)

		df_eval = calc_metrics(args, df_preds)
		df_eval['CLMBR'] = 'PT' if args.pretrain_group == 'pretrained' else 'FT'
		df_eval['task'] = task
		df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)

