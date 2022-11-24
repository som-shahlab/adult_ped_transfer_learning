#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/jlemmon/conl
# script dir
cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

#mkdir -p ../logs/clmbr_eval

LEARN_RATES=(0.01 0.001 0.0001 0.00001)
PT_GROUPS=("ad" "ped") #("all" "mix" "ad" "ped")
TASKS=("aki2_label" "hg_label" "np_500_label" "np_1000_label") #("hospital_mortality" "sepsis" "LOS_7" "readmission_30" "aki1_label" "aki2_label" "hg_label" "np_500_label" "np_1000_label")
TRAIN_OVERWRITE='False'
FEATURIZE_OVERWRITE='False'
EARLY_STOPPING='True'

N_GPU=1
GPU_NUM_START=0
N_JOBS=1

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#PT_GROUPS[@]}
N_LEARN_RATES=${#LEARN_RATES[@]}
N_TASKS=${#TASKS[@]}

for ((r=0; r<$N_TASKS; r++)); do
	for (( t=0; t<$N_GROUPS; t++ )); do
		for (( i=0; i<$N_LEARN_RATES; i++ )); do
			python -u test_clmbr.py \
				--pretrain_group=${PT_GROUPS[$t]} \
				--lr=${LEARN_RATES[$i]} \
				--task=${TASKS[$r]}
		done
	done
done
