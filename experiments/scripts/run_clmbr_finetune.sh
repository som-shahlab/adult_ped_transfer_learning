#!/bin/bash

# conda env
#source activate /local-scratch/nigam/envs/jlemmon/conl
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness
# script dir
cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

#mkdir -p ../logs/clmbr_pretrain

ENCODERS=("gru")
PT_GROUPS=("all") # ("all" "ad")
LEARN_RATES=(0.0001) #(0.0001 0.00001)
L2_VALS=("0") #(0 0.01)
TRAIN_OVERWRITE='False'
FEATURIZE_OVERWRITE='False'
EARLY_STOPPING='True'

GPU="7"
SIZE=800
EPOCHS=100
BATCH_SIZE=4000
DROPOUT="0"


# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#PT_GROUPS[@]}
N_ENCODERS=${#ENCODERS[@]}
N_LEARN_RATES=${#LEARN_RATES[@]}
N_L2_VALS=${#L2_VALS[@]}

export CUDA_VISIBLE_DEVICES="$GPU"

for (( t=0; t<$N_GROUPS; t++ )); do
    for (( i=0; i<$N_ENCODERS; i++ )); do
		for (( l=0; l<$N_LEARN_RATES; l++ )) do

			for (( r=0; r<$N_L2_VALS; r++ )) do
				python -u finetune_clmbr.py \
					--cohort_type=${PT_GROUPS[$t]} \
					--encoder=${ENCODERS[$i]} \
					--lr=${LEARN_RATES[$l]} \
					--l2=${L2_VALS[$r]} \
					--dropout="$DROPOUT" \
					--epochs="$EPOCHS" \
					--batch_size="$BATCH_SIZE" \
					--size="$SIZE" 
			done
		done
	done
done
