#!/bin/bash
#SBATCH --job-name=extract_features_starr_lite
#SBATCH --output=/home/spfohl/slurm_out/extract_features_starr_lite.out
#SBATCH --error=/home/spfohl/slurm_out/extract_features_starr_lite.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH -c 20
#SBATCH -p normal,gpu

source activate prediction_utils

DATASET="starr_omop_cdm5_deid_1pcent_latest"
RS_DATASET="temp_dataset"
COHORT_NAME="starr_1pcent_20200525"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_1pcent_20200525"
GCLOUD_STORAGE_PATH="gs://feature_extraction_exports/cohorts/admissions/starr_1pcent_20200525"

FEATURES_DATASET="temp_dataset"
GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-rit-phi-starr-prod"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"
FEATURES_PREFIX="features_"$USER
INDEX_DATE_FIELD='admit_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field=$INDEX_DATE_FIELD \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --analysis_ids="measurement_bin_dt" "age_group" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite