#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/jlemmon/conl

# script dir
cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

mkdir -p ../logs/clmbr_pretrain

ENCODERS=("gru")
PT_GROUPS=("mix" "ad" "ped")
TRAIN_OVERWRITE='False'
FEATURIZE_OVERWRITE='False'
EARLY_STOPPING='True'

N_GPU=2
GPU_NUM_START=2
N_JOBS=8

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#PT_GROUPS[@]}
N_ENCODERS=${#ENCODERS[@]}
N_CONSTRAINTS=${#CONSTRAINED_PRETRAINING[@]}

for (( t=0; t<$N_GROUPS; t++ )); do
    for (( i=0; i<$N_ENCODERS; i++ )); do
		python -u train_clmbr.py \
			--pretrain_group=${PT_GROUPS[$t]} \
			--encoder=${ENCODERS[$i]} \
			--n_gpu="$N_GPU" \
			--n_jobs="$N_JOBS" \
			--gpu_num_start="$GPU_NUM_START" \
    done
done