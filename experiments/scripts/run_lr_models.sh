#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.lemmon@sickkids.ca
#SBATCH --time=5-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=tl_lr_models
#SBATCH --nodes=1 
#SBATCH -n 16 #number of cores to reserve, default is 1
#SBATCH --mem=50000 # in MegaBytes. default is 8 GB
#SBATCH --partition=shahlab # Partition allocated for the lab
#SBATCH --error=logs/error-sbatchjob.%J.err
#SBATCH --output=logs/out-sbatchjob.%J.out

source activate /labs/shahlab/envs/jlemmon/conl

cd /labs/shahlab/projects/jlemmon/transfer_learning/experiments/scripts

# make log folders if not exist
#mkdir -p ../logs/train_lr
#mkdir -p ../logs/test_lr

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
COHORT_TYPES=("pediatric" "adult")
FEAT_GROUPS=("shared" "pediatric")
TASKS=('hospital_mortality' 'LOS_7' 'readmission_30' 'icu_admission' 'aki1_label' 'aki2_label' 'hg_label' 'np_500_label' 'np_1000_label')
N_BOOT=1000

# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_TASKS jobs in parallel
N_JOBS=1

# whether to re-run 
TUNE_OVERWRITE='False'
TRAIN_OVERWRITE='False'
EVAL_OVERWRITE='False'

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_COHORTS=${#COHORT_TYPES[@]}
N_TASKS=${#TASKS[@]}
N_GROUPS=${#FEAT_GROUPS[@]}

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {

    # Training LR models
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_COHORTS; ij++ )); do
    	for (( g=0; g<$N_GROUPS; g++)); do
        	for (( t=0; t<$N_TASKS; t++ )); do

	            python -u train_lr.py \
	                --task=${TASKS[$t]} \
	                --cohort_type="${COHORT_TYPES[$ij]" \
	                --feat_group="${FEAT_GROUPS[$g]}" #\
	                #>> "../logs/train_lr/${1:2:2}-${1: -2}-${TASKS[$t]}-$JOB_ID" &

	            let k+=1
	            [[ $((k%N_TASKS)) -eq 0 ]] && wait
	        done
        done
    done
    
    # evaluate models
    # executes $N_TASK jobs in parallel
    python -u test_lr.py #\
        #>> "../logs/test_lr/${1:2:2}-${1: -2}-${TASKS[$t]}-$JOB_ID" &

}

## -----------------------------------------------------------
## ----------------------- execute job -----------------------
## -----------------------------------------------------------
# execute $N_JOBS pipes in parallel
c=0
        
pipe  &

let c+=1
[[ $((c%N_JOBS)) -eq 0 ]] && wait


echo "DONE"