#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/jlemmon/conl

cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts

PROJECT="som-nero-nigam-starr"
DATASET="starr_omop_cdm5_deid_2022_08_01_no_"
TARGET_FPATH="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bq_downloads/"

TABLE_NAMES=("condition_occurrence" "measurement" "device_exposure" "procedure_occurrence" "drug_exposure")

N_TABLES=${#TABLE_NAMES[@]}

for (( i=0; i<$N_TABLES; i++ )); do
	python download_bq.py \
			--project=$PROJECT \
			--dataset="$DATASET${TABLE_NAMES[$i]}" \
			--target_fpath=$TARGET_FPATH
	done