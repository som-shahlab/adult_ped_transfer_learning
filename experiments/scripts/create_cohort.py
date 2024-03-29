import configargparse as argparse
import pandas as pd
import numpy as np
import os
from prediction_utils.cohorts.admissions.cohort import BQAdmissionRollupCohort
from prediction_utils.cohorts.admissions.cohort import BQAdmissionOutcomeCohort
from prediction_utils.cohorts.admissions.cohort import BQFilterInpatientCohort
from prediction_utils.util import patient_split_cv

parser = argparse.ArgumentParser()

parser.add_argument(
	"--dataset", type=str, default="starr_omop_cdm5_deid_2022_08_01"
)
parser.add_argument("--rs_dataset", type=str, default="jlemmon_explore")
parser.add_argument("--et_dataset", type=str, default="jlemmon_explore")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str, default="som-nero-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="som-nero-nigam-starr")
parser.add_argument(
    "--rs_dataset_project", type=str, default="som-nero-nigam-starr"
)
parser.add_argument("--cohort_name", type=str, default="tl_admission_rollup_temp")
parser.add_argument(
    "--cohort_name_labeled", type=str, default="tl_admission_rollup_labeled_temp"
)
parser.add_argument(
    "--cohort_name_filtered", type=str, default="tl_admission_rollup_filtered_temp"
)
parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)
parser.add_argument(
    "--no_has_birth_datetime", dest="has_birth_datetime", action="store_false"
)
parser.add_argument("--df_name", type=str, default="cohort")
parser.add_argument(
    "--data_path",
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/transfer_learning/experiments/data/",
)
parser.add_argument(
    "--min_stay_hour",
    type=int,
    default=0,
)
parser.add_argument(
    "--min_pat_age",
    type=int,
    default=-1,
)
parser.add_argument(
    "--filter_query",
    type=str,
    default="",
)
parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)
parser.set_defaults(has_birth_datetime=True)

if __name__ == "__main__":

	args = parser.parse_args()
	cohort = BQAdmissionRollupCohort(**args.__dict__)
	print(cohort.get_create_query())
	cohort.create_cohort_table()

	cohort_labeled = BQAdmissionOutcomeCohort(**args.__dict__)
	print(cohort_labeled.get_create_query())
	cohort_labeled.create_cohort_table()

	cohort_filtered = BQFilterInpatientCohort(**args.__dict__)
	print(cohort_filtered.get_create_query())
	cohort_filtered.create_cohort_table()
	cohort_df = pd.read_gbq(
		"""
			SELECT *
			FROM `{rs_dataset_project}.{rs_dataset}.{cohort_name_filtered}`
		""".format(
			**args.__dict__
		),
		dialect='standard',
	)
	cohort_df = patient_split_cv(
		cohort_df, patient_col="person_id", test_frac=0.1, nfold=10, seed=386
	)
	

	cohort_df['death_date'] = pd.to_datetime(cohort_df['death_date'])
	
	conditions = [cohort_df['age_group'] == '<18']
	outputs = [0]
	
	cohort_df['adult_at_admission'] = np.select(conditions, outputs, 1)
	
	
	cohort_path = os.path.join(args.data_path, "cohort")
	os.makedirs(cohort_path, exist_ok=True)
	cohort_df.to_parquet(
		os.path.join(cohort_path, f"{args.df_name}.parquet"), engine="pyarrow", index=False,
	)