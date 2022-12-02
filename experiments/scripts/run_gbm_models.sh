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

#source activate /local-scratch/nigam/envs/jlemmon/conl 
source activate /home/jlemmon/.conda/envs/tl

#cd /local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/scripts 
cd /labs/shahlab/projects/jlemmon/transfer_learning/experiments/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
COHORT_TYPES=("pediatric" "adult")
FEAT_GROUPS=("shared" "pediatric" "adult")
TASKS=('hospital_mortality' 'sepsis' 'LOS_7' 'readmission_30' 'hyperkalemia_lab_mild', 'hyperkalemia_lab_moderate', 'hyperkalemia_lab_severe', 'hyperkalemia_lab_abnormal', 'hypoglycemia_lab_mild', 'hypoglycemia_lab_moderate', 'hypoglycemia_lab_severe', 'hypoglycemia_lab_abnormal', 'neutropenia_lab_mild', 'neutropenia_lab_moderate', 'neutropenia_lab_severe', 'hyponatremia_lab_mild', 'hyponatremia_lab_moderate', 'hyponatremia_lab_severe', 'hyponatremia_lab_abnormal', 'aki_lab_aki1', 'aki_lab_aki2', 'aki_lab_aki3', 'aki_lab_abnormal', 'anemia_lab_mild', 'anemia_lab_moderate', 'anemia_lab_severe', 'anemia_lab_abnormal', 'thrombocytopenia_lab_mild', 'thrombocytopenia_lab_moderate', 'thrombocytopenia_lab_severe', 'thrombocytopenia_lab_abnormal', 'hypoglycemia_dx', 'aki_dx', 'anemia_dx', 'hyperkalemia_dx', 'hyponatremia_dx', 'thrombocytopenia_dx', 'neutropenia_dx')
MODELS=('gbm' 'gbm_ft')
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

	echo "FINETUNING MODELS..."
	# finetune models
	# executes $N_TASK jobs in parallel
	local k=0
	for (( t=0; t<$N_TASKS; t++ )); do
		python -u finetune_gbm.py \
			   --task=${TASKS[$t]} \
			   --bin_path="$1" \
			   --cohort_path="$2" \
			   --hparam_path="$3" \
			   --model_path="$4" \
			   --result_path="$5" #\
		let k+=1
		[[ $((k%N_TASKS)) -eq 0 ]] && wait
	 done
	 
	echo "EVALUATING MODELS..."
	#evaluate models
	#executes $N_TASK jobs in parallel
	local k=0
	for (( m=0; m<$N_MODELS; m++)); do
		for (( ij=0; ij<$N_COHORTS; ij++ )); do
			for (( t=0; t<$N_TASKS; t++ )); do
				python -u test_gbm.py \
					   --task=${TASKS[$t]} \
					   --model=${MODELS[$m]} \
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
	done

}

## -----------------------------------------------------------
## ----------------------- execute job -----------------------
## -----------------------------------------------------------
# execute $N_JOBS pipes in parallel
c=0
        
pipe "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/bin_features" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/data/cohort" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/hyperparams" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/models" "/labs/shahlab/projects/jlemmon/transfer_learning/experiments/artifacts/results"  &

#pipe "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/bin_features" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/cohort" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/hyperparams" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/models" "/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/artifacts/results"  &

let c+=1
[[ $((c%N_JOBS)) -eq 0 ]] && wait


echo "DONE"
