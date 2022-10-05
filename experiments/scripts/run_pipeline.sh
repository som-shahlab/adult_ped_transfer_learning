#!/bin/bash
source activate /local-scratch/nigam/envs/jlemmon/conl
echo "Running cohort creation..."
python create_cohort.py
echo "Running cohort splitting..."
python split_cohort.py
echo "Running feature extraction..."
python extract_features.py --exclude_analysis_ids "note_nlp" "note_nlp_dt" "note_nlp_delayed" --time_bins "-365000" "-365" "-30" "0" --binary --featurize --no_cloud_storage --merge_features --create_sparse --no_create_parquet --overwrite
echo "Splitting feature set..."
python split_bin.py
echo "Pipeline finished. Can now run feature_explore.ipynb"
