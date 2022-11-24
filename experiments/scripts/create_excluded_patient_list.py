import os
import argparse
import random

import numpy as np
import pandas as pd

from ehr_ml.clmbr import convert_patient_data
from ehr_ml.extension.timeline import TimelineReader

parser = argparse.ArgumentParser(description='Create exclude patient list')

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
	"--cohort_fname",
	type=str,
	default="cohort_split.parquet",
)

parser.add_argument(
	"--excluded_patient_ids_fpath",
	type=str,
	default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/held_out_patients/",
)

parser.add_argument(
	"--excluded_patient_ids_fname",
	type=str,
	default="excluded_patient_ids",
)

parser.add_argument(
	"--included_patient_ids_fpath",
	type=str,
	default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/pretrain_cohort/",
)

parser.add_argument(
	"--included_patient_ids_train_fname",
	type=str,
	default="train_patient_ids",
)

parser.add_argument(
	"--included_patient_ids_val_fname",
	type=str,
	default="val_patient_ids",
)

parser.add_argument(
	"--cutoff_date",
	type=str,
	default='2020-01-01'
)

parser.add_argument(
	"--seed",
	type=int,
	default=44,
)

def read_file(filename, columns=None, **kwargs):

	load_extension = os.path.splitext(filename)[-1]
	if load_extension == ".parquet":
		return pd.read_parquet(filename, columns=columns,**kwargs)
	elif load_extension == ".csv":
		return pd.read_csv(filename, usecols=columns, **kwargs)

if __name__ == "__main__":

	args = parser.parse_args()

	og_df=read_file(
		os.path.join(
			args.cohort_fpath,
			args.cohort_fname
		)
	)

# 	# patient IDs excluded from CLMBR pretraining
# 	# this is for the mixed pretraining
# 	ex_df=og_df.query("fold_id==['val','test']")[['person_id','admit_date','discharge_date']]
# 	ex_df = ex_df[ex_df['person_id'] != 86281596]
# 	ex_df['date']=pd.to_datetime(ex_df['admit_date']).dt.date

# 	ex_df.to_csv(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_mix.csv",
# 		),
# 		index=False
# 	)

# 	ehr_ml_patient_ids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		ex_df['person_id'], 
# 		ex_df['date']
# 	)

# 	with open(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_mix.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in ehr_ml_patient_ids:
# 			f.write("%d\n" % pid)


# 	# patient IDs included for CLMBR pretraining
# 	in_df=og_df.query("fold_id not in ['val','test']")[['person_id','admit_date','discharge_date']]
# 	in_df = in_df[~in_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]
# 	in_df['date']=pd.to_datetime(in_df['admit_date']).dt.date

# 	pids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		in_df['person_id'], 
# 		in_df['date']
# 	)

# 	print(len(pids))
# 	train_end = round(0.8*len(pids))

# 	random.Random(args.seed).shuffle(pids)

# 	train_pids = pids[:train_end]
# 	val_pids = pids[train_end:]

# 	os.makedirs(args.included_patient_ids_fpath,exist_ok=True)

# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_train_fname}_mix.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in train_pids:
# 			f.write("%d\n" % pid)


# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_val_fname}_mix.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in val_pids:
# 			f.write("%d\n" % pid)

# 	# patient IDs excluded from CLMBR pretraining
# 	# this is for the pediatric only pretraining
# 	ex_df = og_df.query("fold_id==['val','test'] | adult_at_admission==1")[['person_id','admit_date','discharge_date']]
# 	ex_df = ex_df[ex_df['person_id'] != 86281596]
# 	ex_df['date']=pd.to_datetime(ex_df['admit_date']).dt.date

# 	ex_df.to_csv(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_ped.csv",
# 		),
# 		index=False
# 	)

# 	ehr_ml_patient_ids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		ex_df['person_id'], 
# 		ex_df['date']
# 	)

# 	with open(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_ped.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in ehr_ml_patient_ids:
# 			f.write("%d\n" % pid)


# 	# patient IDs included for CLMBR pretraining
# 	in_df=og_df.query("fold_id not in ['val','test'] and adult_at_admission==0")[['person_id','admit_date','discharge_date']]
# 	in_df = in_df[~in_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]
# 	in_df['date']=pd.to_datetime(in_df['admit_date']).dt.date
# 	# timelines = TimelineReader(os.path.join(args.extracts_fpath, "extract.db"))

# 	pids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		in_df['person_id'], 
# 		in_df['date']
# 	)
# 	print(len(pids))
# 	train_end = round(0.8*len(pids))

# 	random.Random(args.seed).shuffle(pids)

# 	train_pids = pids[:train_end]
# 	val_pids = pids[train_end:]


# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_train_fname}_ped.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in train_pids:
# 			f.write("%d\n" % pid)


# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_val_fname}_ped.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in val_pids:
# 			f.write("%d\n" % pid)

# 	# patient IDs excluded from CLMBR pretraining
# 	# this is for the adult only pretraining
# 	ex_df=og_df.query("fold_id==['val','test'] | adult_at_admission==0")[['person_id','admit_date','discharge_date']]
# 	ex_df = ex_df[~ex_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]
# 	ex_df['date']=pd.to_datetime(ex_df['admit_date']).dt.date

# 	ex_df.to_csv(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_ad.csv",
# 		),
# 		index=False
# 	)

# 	ehr_ml_patient_ids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		ex_df['person_id'], 
# 		ex_df['date']
# 	)

# 	with open(
# 		os.path.join(
# 			args.excluded_patient_ids_fpath,
# 			f"{args.excluded_patient_ids_fname}_ad.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in ehr_ml_patient_ids:
# 			f.write("%d\n" % pid)


# 	# patient IDs included for CLMBR pretraining
# 	in_df=og_df.query("fold_id not in ['val','test'] and adult_at_admission==1")[['person_id','admit_date','discharge_date']]
# 	# in_df = ex_df[~ex_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]
# 	in_df['date']=pd.to_datetime(in_df['admit_date']).dt.date
# 	# timelines = TimelineReader(os.path.join(args.extracts_fpath, "extract.db"))

# 	pids, day_indices = convert_patient_data(
# 		args.extracts_fpath, 
# 		in_df['person_id'], 
# 		in_df['date']
# 	)
	
# 	# pids = timelines.get_patient_ids()
# 	# pids = list(set(pids).difference(set(ehr_ml_patient_ids)))
# 	print(len(pids))
# 	train_end = round(0.8*len(pids))

# 	random.Random(args.seed).shuffle(pids)

# 	train_pids = pids[:train_end]
# 	val_pids = pids[train_end:]


# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_train_fname}_ad.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in train_pids:
# 			f.write("%d\n" % pid)


# 	with open(
# 		os.path.join(
# 			args.included_patient_ids_fpath,
# 			f"{args.included_patient_ids_val_fname}_ad.txt"
# 		), 
# 		"w"
# 	) as f:

# 		for pid in val_pids:
# 			f.write("%d\n" % pid)

	# patient IDs excluded from CLMBR pretraining
	# this is for the all patient pretraining
	ex_df=og_df.query("fold_id==['val','test'] | adult_at_admission==0")[['person_id','admit_date','discharge_date']]
	ex_df = ex_df[~ex_df['person_id'].isin([86281596,72463221, 31542622, 30046470])]
	ex_df['date']=pd.to_datetime(ex_df['admit_date']).dt.date

	ex_df.to_csv(
		os.path.join(
			args.excluded_patient_ids_fpath,
			f"{args.excluded_patient_ids_fname}_all.csv",
		),
		index=False
	)

	ehr_ml_patient_ids, day_indices = convert_patient_data(
		args.extracts_fpath, 
		ex_df['person_id'], 
		ex_df['date']
	)

	with open(
		os.path.join(
			args.excluded_patient_ids_fpath,
			f"{args.excluded_patient_ids_fname}_all.txt"
		), 
		"w"
	) as f:

		for pid in ehr_ml_patient_ids:
			f.write("%d\n" % pid)


	# patient IDs included for CLMBR pretraining
	timelines = TimelineReader(os.path.join(args.extracts_fpath, "extract.db"))

	pids = timelines.get_patient_ids()
	pids = list(set(pids).difference(set(ehr_ml_patient_ids)))
	print(len(pids))
	train_end = round(0.8*len(pids))

	random.Random(args.seed).shuffle(pids)

	train_pids = pids[:train_end]
	val_pids = pids[train_end:]


	with open(
		os.path.join(
			args.included_patient_ids_fpath,
			f"{args.included_patient_ids_train_fname}_all.txt"
		), 
		"w"
	) as f:

		for pid in train_pids:
			f.write("%d\n" % pid)


	with open(
		os.path.join(
			args.included_patient_ids_fpath,
			f"{args.included_patient_ids_val_fname}_all.txt"
		), 
		"w"
	) as f:

		for pid in val_pids:
			f.write("%d\n" % pid)