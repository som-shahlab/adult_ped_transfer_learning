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
    
    df=read_file(
        os.path.join(
            args.cohort_fpath,
            args.cohort_fname
        )
    )
    
    
    # patient IDs excluded from CLMBR pretraining
    df=df.query("fold_id==['val','test']")[['person_id','admit_date','discharge_date']]
    df['date']=pd.to_datetime(df['admit_date']).dt.date
    
    df.to_csv(
        os.path.join(
            args.excluded_patient_ids_fpath,
            f"{args.excluded_patient_ids_fname}.csv",
        ),
        index=False
    )
    
    ehr_ml_patient_ids, day_indices = convert_patient_data(
        args.extracts_fpath, 
        df['person_id'], 
        df['date']
    )
    
    with open(
        os.path.join(
            args.excluded_patient_ids_fpath,
            f"{args.excluded_patient_ids_fname}.txt"
        ), 
        "w"
    ) as f:
        
        for pid in ehr_ml_patient_ids:
            f.write("%d\n" % pid)
            
            
    # patient IDs included for CLMBR pretraining
    timelines = TimelineReader(os.path.join(args.extracts_fpath, "extract.db"))
    
    pids = timelines.get_patient_ids()
    pids = list(set(pids).difference(set(ehr_ml_patient_ids)))
    
    train_end = round(0.8*len(pids))

    random.Random(args.seed).shuffle(pids)
    
    train_pids = pids[:train_end]
    val_pids = pids[train_end:]
    
    os.makedirs(args.included_patient_ids_fpath,exist_ok=True)
    
    with open(
        os.path.join(
            args.included_patient_ids_fpath,
            f"{args.included_patient_ids_train_fname}.txt"
        ), 
        "w"
    ) as f:
        
        for pid in train_pids:
            f.write("%d\n" % pid)
            
    
    with open(
        os.path.join(
            args.included_patient_ids_fpath,
            f"{args.included_patient_ids_val_fname}.txt"
        ), 
        "w"
    ) as f:
        
        for pid in val_pids:
            f.write("%d\n" % pid)