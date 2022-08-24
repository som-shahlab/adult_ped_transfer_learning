"""
Create excluded patient list for cohort creation.
"""
import os
import argparse

import numpy as np
import pandas as pd

from ehr_ml.clmbr import convert_patient_data

parser = argparse.ArgumentParser(description='Create exclude patient list')

parser.add_argument(
	'--extracts_fpath', 
	type=str,
	default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/extracts/20210723",
)

parser.add_argument(
	'--cohort_fpath', 
	type=str,
	default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort",
)

parser.add_argument(
	"--cohort_fname",
	type=str,
	default="cohort_split.parquet"
)

parser.add_argument(
	"--target_fpath",
	type=str,
	default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/held_out_patients/"
)

parser.add_argument(
	"--target_fname",
	type=str,
	default="excluded_patient_ids"
)

parser.add_argument(
	"--pediatric",
	type=int,
	default=0
)

def read_file(filename, columns=None, **kwargs):

	load_extension = os.path.splitext(filename)[-1]
	if load_extension == ".parquet":
		return pd.read_parquet(filename, columns=columns,**kwargs)
	elif load_extension == ".csv":
		return pd.read_csv(filename, usecols=columns, **kwargs)

if __name__ == "__main__":

	args = parser.parse_args()
	
	os.makedirs(args.target_fpath, exist_ok=True)
	
	df=read_file(
		os.path.join(
			args.cohort_fpath,
			args.cohort_fname
		)
	)

	df=df.query("fold_id==['val','test']")[['person_id','admit_date','discharge_date', 'age_group']]
	if args.pediatric == 1:
		df=df.query("age_group=='<18'")
	else:
		df=df.query("age_group==['[18-30)', '[45-55)', '[30-45)', '[55-65)','[75-91)', '[65-75)']")
	df['date']=pd.to_datetime(df['admit_date']).dt.date

	# save csv
	df.to_csv(
		os.path.join(
			args.target_fpath,
			f"{args.target_fname}.csv",
		),
		index=False
	)

	# convert patient data
	ehr_ml_patient_ids, day_indices = convert_patient_data(
		args.extracts_fpath, 
		df['person_id'], 
		df['date']
	)

	# write to text file
	with open(
		os.path.join(
			args.target_fpath,
			f"{args.target_fname}_{'pediatric' if args.pediatric==1 else 'adult'}.txt"
		), 
		"w"
	) as f:

		for pid in ehr_ml_patient_ids:
			f.write("%d\n" % pid)