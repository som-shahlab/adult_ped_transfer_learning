import os
import json
import argparse
import yaml
import pickle
from datetime import datetime

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import convert_patient_data

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

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
    '--labelled_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/clmbr_labelled_data",
    help='Base path for labelled data directory'
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
    '--cohort_dtype',
    type=str,
    default='parquet',
    help='Data type for cohort file.'
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=4,
    help="number of threads"
)

parser.add_argument(
	'--seed',
	type=int,
	default=44
)

if __name__ == '__main__':
    
	args = parser.parse_args()

	np.random.seed(args.seed)

	tasks = ['hospital_mortality', 'sepsis', 'LOS_7','readmission_30','icu_admission','aki1_label','aki2_label','hg_label','np_500_label','np_1000_label']

	if args.cohort_dtype == 'parquet':
		og_df = pd.read_parquet(os.path.join(args.cohort_fpath, "cohort_split.parquet"))
	else:
		og_df = pd.read_csv(os.path.join(args.cohort_fpath, "cohort_split.csv"))

	train_start_date = pd.to_datetime(args.train_start_date)
	train_end_date = pd.to_datetime(args.train_end_date)
	val_start_date = pd.to_datetime(args.val_start_date)
	val_end_date = pd.to_datetime(args.val_end_date)
	test_start_date = pd.to_datetime(args.test_start_date)
	test_end_date = pd.to_datetime(args.test_end_date)

	og_df = og_df.assign(date = pd.to_datetime(og_df['admit_date']).dt.date)
	og_df = og_df[~og_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]

	for ct in ['mix', 'ped', 'ad']:
		ehr_ml_patient_ids = {}
		prediction_ids = {}
		day_indices = {}
		labels = {}
		features = {}
		
		if ct == 'ped':
			dataset = og_df.query('adult_at_admission!=1')
		elif ct == 'ad':
			dataset = og_df.query('adult_at_admission==1')
		else:
			dataset = og_df
		for task in tasks:

			ehr_ml_patient_ids[task] = {}
			prediction_ids[task] = {}
			day_indices[task] = {}
			labels[task] = {}
			features[task] = {}

			if task == 'readmission':
				index_year = 'discharge_year'
			else:
				index_year = 'admission_year'

			for fold in ['train', 'val', 'test']:
				print(f'Creating task {task} fold {fold} id list')

				if fold == 'train':
					df = dataset.query(f"{task}_fold_id!=['test','val','ignore']")
					print('train end:', train_end_date)
					mask = ((df['date'] >= train_start_date) & (df['date'] <= train_end_date))
				elif fold == 'val':
					df = dataset.query(f"{task}_fold_id==['val']")
					print('val start:', val_start_date, 'val end:', val_end_date)
					mask = ((df['date'] >= val_start_date) & (df['date'] <= val_end_date))
				else:
					df = dataset.query(f"{task}_fold_id==['test']")
					print('test start:', test_start_date, 'test end:', test_end_date)
					mask = ((df['date'] >= test_start_date) & (df['date'] <= test_end_date))
				df = df.loc[mask].reset_index()
				print(min(df['date']))
				print(max(df['date']))
				ehr_ml_patient_ids[task][fold], day_indices[task][fold] = convert_patient_data(args.extract_path, df['person_id'], 
																							   df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date)
				labels[task][fold] = df[task]
				prediction_ids[task][fold]=df['prediction_id']


				assert(
					len(ehr_ml_patient_ids[task][fold]) ==
					len(labels[task][fold]) == 
					len(prediction_ids[task][fold])
				)


				task_path = f'{args.labelled_fpath}/{task}/{ct}'
				print(task_path)
				if not os.path.exists(task_path):
					os.makedirs(task_path)
				df_ehr_pat_ids = pd.DataFrame(ehr_ml_patient_ids[task][fold])
				df_ehr_pat_ids.to_csv(f'{task_path}/ehr_ml_patient_ids_{fold}.csv', index=False)

				df_prediction_ids = pd.DataFrame(prediction_ids[task][fold])
				df_prediction_ids.to_csv(f'{task_path}/prediction_ids_{fold}.csv', index=False)

				df_day_inds = pd.DataFrame(day_indices[task][fold])
				df_day_inds.to_csv(f'{task_path}/day_indices_{fold}.csv', index=False)

				df_labels = pd.DataFrame(labels[task][fold])
				df_labels.to_csv(f'{task_path}/labels_{fold}.csv', index=False)

                
            
        
    