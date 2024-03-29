import os
import torch
import bisect
import datetime
import argparse
import sys
import re
import pickle
import gzip
import pdb

import numpy as np
import pandas as pd
import ehr_ml.clmbr

from ehr_ml.clmbr import convert_patient_data 
from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
	description = "Hyperparameter sweep"
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
	"--cohort_path",
	type = str,
	default = "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort",
	help = "path to save cohort"
)

parser.add_argument(
	"--cohort_id",
	type = str,
	default = "all",
	help = "cohort clmbr model was pretrained on"
)

parser.add_argument(
	"--train_type",
	type = str,
	default = "finetuned",
	help = "how clmbr model was pretrained"
)

parser.add_argument(
	"--device",
	type=str,
	default='cuda:0',
	help='CUDA device',
)

parser.add_argument(
	"--use_pretrained",
	type = str2bool,
	default = "false",
	help = "whether to use a pretrained model to generate constrained feats",
)

parser.add_argument(
	"--overwrite",
	type = str2bool,
	default = "false",
	help = "whether to overwrite existing artifacts",
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

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":

	args = parser.parse_args()

	tasks = ['hospital_mortality','sepsis','LOS_7','readmission_30','hyperkalemia_lab_mild_label','hyperkalemia_lab_moderate_label','hyperkalemia_lab_severe_label','hyperkalemia_lab_abnormal_label','hypoglycemia_lab_mild_label','hypoglycemia_lab_moderate_label','hypoglycemia_lab_severe_label','hypoglycemia_lab_abnormal_label','neutropenia_lab_mild_label','neutropenia_lab_moderate_label','neutropenia_lab_severe_label','hyponatremia_lab_mild_label','hyponatremia_lab_moderate_label','hyponatremia_lab_severe_label','hyponatremia_lab_abnormal_label','aki_lab_aki1_label','aki_lab_aki2_label','aki_lab_aki3_label','aki_lab_abnormal_label','anemia_lab_mild_label','anemia_lab_moderate_label','anemia_lab_severe_label','anemia_lab_abnormal_label','thrombocytopenia_lab_mild_label','thrombocytopenia_lab_moderate_label','thrombocytopenia_lab_severe_label','thrombocytopenia_lab_abnormal_label']


	lr = '0.0001'
	for percent in list(range(5,100,5)):
		
		cohort = read_file(
			os.path.join(
				args.cohort_path,
				f"cohort_split_no_nb_constrain_{percent}.parquet"
			),
			engine='pyarrow'
		)
		cohort = cohort[~cohort['person_id'].isin([86281596,72463221, 31542622, 30046470])]
		if args.use_pretrained:
			clmbr_model_path=os.path.join(
			args.artifacts_fpath,
			"pretrained",
			"models",
			"ad",
			f"gru_sz_800_do_0_lr_{lr}_l2_0"
		)
		else:
			clmbr_model_path=os.path.join(
				args.artifacts_fpath,
				args.train_type,
				"models",
				args.cohort_id,
				f"gru_sz_800_do_0_lr_{lr}_l2_0",
				f"finetune_model_constrain_{percent}"
			)
		
		print(clmbr_model_path)
		clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)

		cohort_group = 'constrain'
		
		save_dir=os.path.join(
			args.artifacts_fpath,
			args.train_type,
			"features",
			args.cohort_id,
			f"gru_sz_800_do_0_lr_{lr}_l2_0",
			f"constrain_{percent}"
		)
		print(save_dir)
		c_df = cohort.query('adult_at_admission==0')
		c_df = cohort.query('constrain==1 | fold_id=="test"')

		# check if files exist
		if all([
			os.path.exists(f"{save_dir}/{f}") for f in 
			['ehr_ml_patient_ids.gz','prediction_ids.gz','day_indices.gz','labels.gz','features.gz']
		]) and not args.overwrite:

			print("Artifacts exist and args.overwrite is set to False. Skipping...")
			sys.exit()

		elif not all([
			os.path.exists(f"{save_dir}/{f}") for f in 
			['ehr_ml_patient_ids.gz','prediction_ids.gz','day_indices.gz','labels.gz','features.gz']
		]) or args.overwrite: 

			os.makedirs(save_dir,exist_ok=True)

			# read dirs
			ehr_ml_extract_dir=args.extracts_fpath

			ehr_ml_patient_ids={}
			prediction_ids={}
			day_indices={}
			labels={}
			row_indices={}
			features={}
			features['all']={}

			for task in tasks:
				ehr_ml_patient_ids[task]={}
				prediction_ids[task]={}
				day_indices[task]={}
				row_indices[task]={}
				labels[task]={}
				features[task]={}

				if task == 'readmission_30':
					index_year = 'discharge_year'
				else:
					index_year = 'admission_year'

				for fold in ['train','val','test']:

					print(f"Featurizing task {task} fold {fold}")


					if fold=='train':
						if task != 'readmission_30':
							df = c_df.query(
								f"fold_id!=['test','val']"
							).reset_index()
						else:
							df = c_df.query(
								f"{task}_fold_id!=['test','val','ignore']"
							).reset_index()
					elif fold=='val':
						if task != 'readmission_30':
							df = c_df.query(
								f"fold_id==['val']"
							).reset_index()
						else:
							df = c_df.query(
								f"{task}_fold_id==['val']"
							).reset_index()
					elif fold=='test':
						if task != 'readmission_30':
							df = c_df.query(
								f"fold_id==['test']"
							).reset_index()
						else:
							df = c_df.query(
								f"{task}_fold_id==['test']"
							).reset_index()
					
					if task == 'hospital_mortality':
						ehr_ml_patient_id_list, day_indices_list = convert_patient_data( 
							ehr_ml_extract_dir, 
							df['person_id'], 
							df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
						)
					
					ehr_ml_patient_ids[task][fold], day_indices[task][fold] = convert_patient_data( 
						ehr_ml_extract_dir, 
						df.query(f'{task}_fold_id!="ignore"')['person_id'], 
						df.query(f'{task}_fold_id!="ignore"')['admit_date'].dt.date if task!='readmission_30' else df.query(f'{task}_fold_id!="ignore"')['discharge_date'].dt.date
					)
					
					labels[task][fold]=np.array(df.query(f'{task}_fold_id!="ignore"')[task].values).astype(np.int32)
					row_indices[task][fold] = df.query(f'{task}_fold_id!="ignore"').index.values
					prediction_ids[task][fold]=df.query(f'{task}_fold_id!="ignore"')['prediction_id']

					assert (
						len(ehr_ml_patient_ids[task][fold]) == 
						len(labels[task][fold]) == 
						len(prediction_ids[task][fold]))
					
					if task == 'readmission_30':
						features[task][fold] = clmbr_model.featurize_patients(
							ehr_ml_extract_dir, 
							np.array(ehr_ml_patient_ids[task][fold]), 
							np.array(day_indices[task][fold])
						)
					elif task == 'hospital_mortality':
						features['all'][fold] = clmbr_model.featurize_patients(
							ehr_ml_extract_dir, 
							np.array(ehr_ml_patient_id_list), 
							np.array(day_indices_list)
						)
					else:
						continue
			# save artifacts  
			print('Saving ehr ml patient ids...')
			pickle.dump(
				ehr_ml_patient_ids,
				gzip.open(os.path.join(save_dir,'ehr_ml_patient_ids.gz'),'wb')
			)
			print('Saving prediction ids...')
			pickle.dump(
				prediction_ids,
				gzip.open(os.path.join(save_dir,'prediction_ids.gz'),'wb')
			)
			print('Saving day indices...')
			pickle.dump(
				day_indices,
				gzip.open(os.path.join(save_dir,'day_indices.gz'),'wb')
			)
			print('Saving feature row indices...')
			pickle.dump(
				row_indices,
				gzip.open(os.path.join(save_dir,'row_indices.gz'),'wb')
			)
			print('Saving labels...')
			pickle.dump(
				labels,
				gzip.open(os.path.join(save_dir,'labels.gz'),'wb')
			)
			print('Saving features...')
			pickle.dump(
				features,
				gzip.open(os.path.join(save_dir,'features.gz'),'wb')
			)