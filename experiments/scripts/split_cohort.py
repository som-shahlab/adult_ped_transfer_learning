import os
import argparse
import pickle
import joblib
import pdb
import warnings

import pandas as pd
import numpy as np
import itertools

from prediction_utils.util import str2bool
from sklearn.model_selection import KFold
from sepsis_labeler.labeler import SepsisLabeler

warnings.filterwarnings("ignore")

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
	description = "Split cohort stratified by admission year and task labels"
)

parser.add_argument(
	"--cohort_fpath",
	type = str,
	default = "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/",
	help = "path to save cohort"
)


parser.add_argument(
	"--tasks",
	nargs='+',
	type=str ,
	default=['hospital_mortality','sepsis','LOS_7','readmission_30','icu_admission','aki1_label','aki2_label','hg_label','np_500_label','np_1000_label'],
	help="List of tasks to predict"
)

parser.add_argument(
	"--seed",
	type = int,
	default = 44,
	help = "seed for deterministic training"
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
	default=2020,
	help='End date of training ids.'
)

parser.add_argument(
	'--val_start_year',
	type=int,
	default=2008,
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

parser.add_argument(
	'--add_sepsis_label',
	type=str2bool,
	default=False
)

parser.add_argument(
	'--cohort_table_name',
	type=str,
	default="tl_admission_rollup_filtered_temp"

)

#------------------------------------
# Helper funs
#------------------------------------
def read_file(filename, columns=None, **kwargs):
	print(filename)
	load_extension = os.path.splitext(filename)[-1]
	if load_extension == ".parquet":
		return pd.read_parquet(filename, columns=columns,**kwargs)
	elif load_extension == ".csv":
		return pd.read_csv(filename, usecols=columns, **kwargs)

def add_sepsis_label(args, cohort, patient_col='person_id'):
	config_dict = {
		"dataset_project": "som-nero-nigam-starr",
		"rs_dataset_project": "som-nero-nigam-starr",
		"dataset": "starr_omop_cdm5_deid_2022_08_01",
		"rs_dataset": "jlemmon_explore",
		"cohort_name": args.cohort_table_name,
		"ext_flwsht_table":"meas_vals_json",
		"print_query": False,
		"save_to_database":False,
		"replace_cohort":False,
		"pre_existing_cohort":True,
	}
	
	sl = SepsisLabeler(**config_dict)
	#df = sl.create_labels()
	df = sl.create_labelled_cohort()
	df = df[['person_id', 'sepsis', 'sepsis_index_date']]
	return cohort.merge(df, on='person_id', how='left')
	
def split_cohort(
		args,
		df,
		seed,
		patient_col='person_id',
		index_year='admission_year',
		tasks=['hospital_mortality','sepsis','LOS_7','readmission_30','icu_admission', 'aki1_label','aki2_label','hg_label','np_500_label','np_1000_label'],
		val_frac=0.15,
		test_frac=0.15,
		nfold=5
	):

	assert (test_frac > 0.0) & (test_frac < 1.0) & (val_frac < 1.0)

	# Get admission year
	df['admission_year']=df['admit_date'].dt.year
	df['discharge_year']=df['discharge_date'].dt.year

	# Split into train, val, and test
	test = df.query('admission_year >= @args.test_start_year').assign(**{f"fold_id":'test'})

	val = df.query('admission_year >= @args.val_start_year and admission_year <=@args.val_end_year').groupby(['adult_at_admission', 'admission_year']).sample(
		frac=val_frac,
		random_state = seed
	).assign(**{
		f"fold_id":'val'
	})
	train = df.query('admission_year >= @args.train_start_year and admission_year <=@args.train_end_year').drop(index=val.index)
	# train = train.drop(index=test.index)
	train = train.assign(**{
		f"fold_id":'train'
	})

	# split train into kfolds
	kf = KFold(
		n_splits=nfold,
		shuffle=True,
		random_state=seed
	)

	cohort = test
	cohort = pd.concat((test,val))
	adult_at_admission = [0, 1]
	years = train['admission_year'].unique()

	for pair in list(itertools.product(adult_at_admission, years)): 
		itrain = train.query(f"adult_at_admission==@pair[0] and admission_year==@pair[1]")
		c=0
		for _, val_ids in kf.split(itrain[patient_col]):

			cohort = pd.concat((cohort,
				itrain.iloc[val_ids,:].assign(**{
					f"fold_id":str(c)
				}))
			)

	for task in tasks:
		assert(task in cohort.columns)
		cohort[f"{task}_fold_id"]=cohort['fold_id']

		# remove deaths before midnight
		cohort.loc[cohort['death_date']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
		cohort.loc[cohort['death_date']<=cohort['admit_date_midnight'],f'{task}']=np.nan

		# remove discharges before midnight
		cohort.loc[cohort['discharge_date']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
		cohort.loc[cohort['discharge_date']<=cohort['admit_date_midnight'],f'{task}']=np.nan

		if task == 'readmission_30':
			# remove admissions in which the patient died
			cohort.loc[cohort['hospital_mortality']==1,f'{task}_fold_id']='ignore'
			cohort.loc[cohort['hospital_mortality']==1,f'{task}']= np.nan
			# remove re-admissions on the same day
			cohort.loc[cohort['readmission_window']==0,f'{task}_fold_id']='ignore'
			cohort.loc[cohort['readmission_window']==0,f'{task}']= np.nan

		if task == 'icu_admission':
			# remove icu admissions before midnight
			cohort.loc[cohort['icu_start_datetime']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['icu_start_datetime']<=cohort['admit_date_midnight'],f'{task}']=np.nan

		if task == 'aki1_label':
			# remove aki1 before midnight
			cohort.loc[cohort['aki1_creatinine_time']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['aki1_creatinine_time']<=cohort['admit_date_midnight'],f'{task}']=np.nan
		if task == 'aki2_label':
			# remove aki2 before midnight
			cohort.loc[cohort['aki2_creatinine_time']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['aki2_creatinine_time']<=cohort['admit_date_midnight'],f'{task}']=np.nan
		if task == 'hg_label':
			# remove hg before midnight
			cohort.loc[cohort['hg_glucose_time']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['hg_glucose_time']<=cohort['admit_date_midnight'],f'{task}']=np.nan
		if task == 'np_500_label':
			# remove np_500 before midnight
			cohort.loc[cohort['np_500_neutrophils_time']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['np_500_neutrophils_time']<=cohort['admit_date_midnight'],f'{task}']=np.nan
		if task == 'np_1000_label':
			# remove np_500 before midnight
			cohort.loc[cohort['np_1000_neutrophils_time']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['np_1000_neutrophils_time']<=cohort['admit_date_midnight'],f'{task}']=np.nan
		if task == 'sepsis':
			# remove sepsis before midnight
			cohort.loc[cohort['sepsis_index_date']<=cohort['admit_date_midnight'],f'{task}_fold_id']='ignore'
			cohort.loc[cohort['sepsis_index_date']<=cohort['admit_date_midnight'],f'{task}']=np.nan
	return cohort.sort_index()

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":

	args = parser.parse_args()
	np.random.seed(args.seed)

	cohort = read_file(
		os.path.join(
			args.cohort_fpath,
			"cohort/cohort.parquet"
		),
		engine='pyarrow'
	)
	
	if args.add_sepsis_label:
		cohort = add_sepsis_label(args, cohort)
	
	# split cohort
	cohort = split_cohort(
		args,
		cohort,
		args.seed
	)

	# save splitted cohort
	cohort.to_parquet(
		os.path.join(
			args.cohort_fpath,
			"cohort/cohort_split.parquet"
		),
		engine="pyarrow",
	)

	print(cohort)