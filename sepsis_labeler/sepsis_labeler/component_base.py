import os
from prediction_utils.extraction_utils.database import BQDatabase

class Component:
	def __init__(self, *args, **kwargs):
		self.config_dict = self.get_config_dict(**kwargs)
		self.db = BQDatabase(**self.config_dict)
	
	def get_component_query(self):
		raise NotImplementedError

	def get_values_query(self):
		raise NotImplementedError

	def get_window_query(self):
		raise NotImplementedError

	def get_rollup_query(self):
		raise NotImplementedError
	
	def create_component_table(self):
		self.db.execute_sql(self.get_component_query())

	def get_defaults(self):
		return {
			"gcloud_project": "som-nero-nigam-starr",
			"dataset_project": None,
			"rs_dataset_project": None,
			"dataset": "starr_omop_cdm5_deid_2022_08_01",
			"rs_dataset": "sepsis_temp_dataset",
			"cohort_name": "sepsis_temp_cohort",
			"ext_flwsht_table": "meas_vals_json",
			"google_application_credentials": os.path.expanduser(
				"~/.config/gcloud/application_default_credentials.json"
			),
			"limit": None,
			"min_stay_hour":0,
			"print_query":False,
			"meas_window_prior":'''DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE)''',
			"meas_window":'''CAST(sepsis_index_date AS DATE) >= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
								  CAST(sepsis_index_date AS DATE) <= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 1 DAY) AS DATE)''',
			"obs_window_prior":'''DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 2 DAY) > CAST(observation_DATETIME AS DATE) AND
								  DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 10 DAY) <= CAST(observation_DATETIME AS DATE)''',
			"obs_window":'''CAST(sepsis_index_date AS DATE) >= CAST(DATETIME_SUB(observation_DATETIME, INTERVAL 2 DAY) AS DATE) AND
								 CAST(sepsis_index_date AS DATE) <= CAST(DATETIME_ADD(observation_DATETIME, INTERVAL 1 DAY) AS DATE)''',
			"drug_window_prior":'''(DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 3 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 4 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 5 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 6 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 7 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 8 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 9 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								   DATE_SUB(CAST(sepsis_index_date AS DATE), INTERVAL 10 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE))''',
			"drug_window":'''(CAST(sepsis_index_date AS DATE) BETWEEN CAST(drug_exposure_start_DATETIME AS DATE) AND CAST(drug_exposure_end_DATETIME AS DATE) OR
								  CAST(DATETIME_ADD(sepsis_index_date, INTERVAL 1 DAY) AS DATE) BETWEEN CAST(drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
							      CAST(DATETIME_SUB(sepsis_index_date, INTERVAL 1 DAY) AS DATE) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
								  CAST(DATETIME_SUB(sepsis_index_date, INTERVAL 2 DAY) AS DATE) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE))'''
		}

	def override_defaults(self, **kwargs):
		return {**self.get_defaults(), **kwargs}

	def get_config_dict(self, **kwargs):
		config_dict = self.override_defaults(**kwargs)

		# Handle special parameters
		config_dict["limit_str"] = (
			"LIMIT {}".format(config_dict["limit"])
			if (
				(config_dict["limit"] is not None)
				and (config_dict["limit"] != "")
				and (config_dict["limit"] != 0)
			)
			else ""
		)
		config_dict["where_str"] = (
			"WHERE DATETIME_DIFF(visit_end_datetime, visit_start_datetime, hour)>{}".format(
				str(config_dict["min_stay_hour"])
			)
			if (
				(config_dict["min_stay_hour"] is not None)
				and (config_dict["min_stay_hour"] != "")
			)
			else ""
		)
		config_dict["dataset_project"] = (
			config_dict["dataset_project"]
			if (
				(config_dict["dataset_project"] is not None)
				and (config_dict["dataset_project"] != "")
			)
			else config_dict["gcloud_project"]
		)
		config_dict["rs_dataset_project"] = (
			config_dict["rs_dataset_project"]
			if (
				(config_dict["rs_dataset_project"] is not None)
				and (config_dict["rs_dataset_project"] != "")
			)
			else config_dict["gcloud_project"]
		)
		return config_dict