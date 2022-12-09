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

python clmbr_adapter.py --eval_cohort 'ped'

echo "DONE"

