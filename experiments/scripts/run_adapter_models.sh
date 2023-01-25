#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.lemmon@sickkids.ca
#SBATCH --time=4-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=tl_adapter_models
#SBATCH --nodes=1 
#SBATCH -n 16 #number of cores to reserve, default is 1
#SBATCH --mem=40000 # in MegaBytes. default is 8 GB
#SBATCH --partition=shahlab # Partition allocated for the lab
#SBATCH --error=/labs/shahlab/projects/jlemmon/transfer_learning/experiments/logs/adapter/error-sbatchjob.%J.err
#SBATCH --output=/labs/shahlab/projects/jlemmon/transfer_learning/experiments/logs/adapter/out-sbatchjob.%J.out

source /home/${USER}/.bashrc
source activate /labs/shahlab/envs/jlemmon/tl

cd /labs/shahlab/projects/jlemmon/transfer_learning/experiments/scripts

python clmbr_adapter.py --eval_cohort 'ad' --overwrite 'true'
python clmbr_adapter.py --eval_cohort 'ped' --overwrite 'true'
python clmbr_adapter.py --train_cohort 'ped' --eval_cohort 'ped' --overwrite 'true'

PERCENTS=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)
N_PERCENTS=${#PERCENTS[@]}

for ((p=0; p <$N_PERCENTS; p++)) do
	python clmbr_adapter.py --constrain 'true' --train_cohort 'constrain' --eval_cohort 'ped' --overwrite 'true' --percent ${PERCENTS[$p]}
done
echo "DONE"

