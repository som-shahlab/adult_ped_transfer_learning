#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/jlemmon/conl
# script dir
cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

#mkdir -p ../logs/clmbr_eval

LEARN_RATES=(0.0001) #(0.01 0.001 0.0001 0.00001)
PT_GROUPS=("all" "ad") #("all" "mix" "ad" "ped")
TASKS=("hospital_mortality" "sepsis" "LOS_7" 'hyperkalemia_lab_mild', 'hyperkalemia_lab_moderate', 'hyperkalemia_lab_severe', 'hyperkalemia_lab_abnormal', 'hypoglycemia_lab_mild', 'hypoglycemia_lab_moderate', 'hypoglycemia_lab_severe', 'hypoglycemia_lab_abnormal', 'neutropenia_lab_mild', 'neutropenia_lab_moderate', 'neutropenia_lab_severe', 'hyponatremia_lab_mild', 'hyponatremia_lab_moderate', 'hyponatremia_lab_severe', 'hyponatremia_lab_abnormal', 'aki_lab_aki1', 'aki_lab_aki2', 'aki_lab_aki3', 'aki_lab_abnormal', 'anemia_lab_mild', 'anemia_lab_moderate', 'anemia_lab_severe', 'anemia_lab_abnormal', 'thrombocytopenia_lab_mild', 'thrombocytopenia_lab_moderate', 'thrombocytopenia_lab_severe', 'thrombocytopenia_lab_abnormal', 'hypoglycemia_dx', 'aki_dx', 'anemia_dx', 'hyperkalemia_dx', 'hyponatremia_dx', 'thrombocytopenia_dx', 'neutropenia_dx')
TRAIN_COHORTS=("ad" "ped")
TEST_COHORTS=("ad" "ped")
TRAIN_TYPES=("pretrained" "finetuned")
TRAIN_OVERWRITE='False'
FEATURIZE_OVERWRITE='False'
EARLY_STOPPING='True'

N_GPU=1
GPU_NUM_START=4
N_JOBS=1

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#PT_GROUPS[@]}
N_LEARN_RATES=${#LEARN_RATES[@]}
N_TASKS=${#TASKS[@]}
N_TR_COHORTS=${#TRAIN_COHORTS[@]}
N_TST_COHORTS=${#TEST_COHORTS[@]}
N_TR_TYPES=${#TRAIN_TYPES[@]}

for ((r=0; r<$N_TASKS; r++)); do
	for ((m=0; m<$N_TR_TYPES; n++)); do
		for (( t=0; t<$N_GROUPS; t++ )); do
			for (( i=0; i<$N_LEARN_RATES; i++ )); do
				for ((k=0; k<$N_TR_COHORTS; k++)); do
					for ((l=0; l<$N_TST_COHORTS; l++)); do
						python -u test_clmbr.py \
							--pretrain_group=${PT_GROUPS[$t]} \
							--train_cohort=${TRAIN_COHORTS[$k]} \
							--test_cohort=${TEST_COHORTS[$l]} \
							--train_type=${TRAIN_TYPES[$m]} \
							--lr=${LEARN_RATES[$i]} \
							--task=${TASKS[$r]}
					done
				done
			done
		done
	done
done
