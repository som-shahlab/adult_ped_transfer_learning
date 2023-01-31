#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/jlemmon/conl

DATA_LOCATION="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bq_downloads/som-nero-nigam-starr.starr_omop_cdm5_deid_2022_08_01_no_"
UMLS_LOCATION="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/2020AB"
GEM_LOCATION="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/gem_mappings"
RXNORM_LOCATION="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/RxNorm"
TARGET_LOCATION="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/extracts/20220801_no_"

TABLE_NAMES=("condition_occurrence" "measurement" "device_exposure" "procedure_occurrence" "drug_exposure")

N_TABLES=${#TABLE_NAMES[@]}
for (( i=0; i<$N_TABLES; i++ )); do
	# target location should not exist
	rm -rf "$TARGET_LOCATION${TABLE_NAMES[$i]}"

	# --use_quotes needed, otherwise throws segmentation error
	ehr_ml_extract_omop \
		"$DATA_LOCATION${TABLE_NAMES[$i]}" \
		"$UMLS_LOCATION" \
		"$GEM_LOCATION"\
		"$RXNORM_LOCATION" \
		"$TARGET_LOCATION${TABLE_NAMES[$i]}" \
		--delimiter ',' \
		--use_quotes 
	done
