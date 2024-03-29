#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.lemmon@sickkids.ca
#SBATCH --time=4-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=tl_gbm_models
#SBATCH --nodes=1 
#SBATCH -n 16 #number of cores to reserve, default is 1
#SBATCH --mem=30000 # in MegaBytes. default is 8 GB
#SBATCH --partition=shahlab # Partition allocated for the lab
#SBATCH --error=/labs/shahlab/projects/jlemmon/transfer_learning/experiments/logs/gbm/error-sbatchjob.%J.err
#SBATCH --output=/labs/shahlab/projects/jlemmon/transfer_learning/experiments/logs/gbm/out-sbatchjob.%J.out

source /home/${USER}/.bashrc
source activate /home/jlemmon/.conda/envs/tl

cd /labs/shahlab/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
COHORT_TYPES=("pediatric" "adult")
FEAT_GROUPS=("shared") # "pediatric" "adult")
TASKS=("hospital_mortality" "sepsis" "LOS_7" "readmission_30" "hyperkalemia_lab_mild_label" "hyperkalemia_lab_moderate_label" "hyperkalemia_lab_severe_label" "hyperkalemia_lab_abnormal_label" "hypoglycemia_lab_mild_label" "hypoglycemia_lab_moderate_label" "hypoglycemia_lab_severe_label" "hypoglycemia_lab_abnormal_label" "neutropenia_lab_mild_label" "neutropenia_lab_moderate_label" "neutropenia_lab_severe_label" "hyponatremia_lab_mild_label" "hyponatremia_lab_moderate_label" "hyponatremia_lab_severe_label" "hyponatremia_lab_abnormal_label" "aki_lab_aki1_label" "aki_lab_aki2_label" "aki_lab_aki3_label" "aki_lab_abnormal_label" "anemia_lab_mild_label" "anemia_lab_moderate_label" "anemia_lab_severe_label" "anemia_lab_abnormal_label" "thrombocytopenia_lab_mild_label" "thrombocytopenia_lab_moderate_label" "thrombocytopenia_lab_severe_label" "thrombocytopenia_lab_abnormal_label")
N_BOOT=1000
# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_TASKS jobs in parallel
N_JOBS=1

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_COHORTS=${#COHORT_TYPES[@]}
N_TASKS=${#TASKS[@]}
N_GROUPS=${#FEAT_GROUPS[@]}
N_MODELS=${#MODELS[@]}

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {

	#Training GBM models
	#executes $N_TASK jobs in parallel
	echo "TRAINING MODELS..."
	local k=0
	for (( ij=0; ij<$N_COHORTS; ij++ )); do
		for (( g=0; g<$N_GROUPS; g++)); do
			for (( t=0; t<$N_TASKS; t++ )); do
				python -u train_gbm.py \
					--task=${TASKS[$t]} \
					--cohort_type=${COHORT_TYPES[$ij]} \
					--feat_group=${FEAT_GROUPS[$g]} \
					--bin_path="$1" \
					--cohort_path="$2" \
					--hparam_path="$3" \
					--model_path="$4" \
					--results_path="$5" #\
				let k+=1
				[[ $((k%N_TASKS)) -eq 0 ]] && wait
			done
		done
	done
	 
	echo "EVALUATING MODELS..."
	#evaluate models
	#executes $N_TASK jobs in parallel
	local k=0
	for (( ij=0; ij<$N_COHORTS; ij++ )); do
		for (( t=0; t<$N_TASKS; t++ )); do
			python -u test_gbm.py \
				   --task=${TASKS[$t]} \
				   --cohort_type=${COHORT_TYPES[$ij]} \
				   --bin_path="$1" \
				   --cohort_path="$2" \
				   --hparam_path="$3" \
				   --model_path="$4" \
				   --result_path="$5" #\
			let k+=1
			[[ $((k%N_TASKS)) -eq 0 ]] && wait
		done
	done

}

## -----------------------------------------------------------
## ----------------------- execute job -----------------------
## -----------------------------------------------------------
# execute $N_JOBS pipes in parallel
c=0
        
pipe "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/bin_features" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/cohort" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/hyperparams" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/models" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/results"  &

#pipe "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bin_features" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/hyperparams" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results"  &

#let c+=1
[[ $((c%10)) -eq 0 ]] && wait


echo "DONE"
