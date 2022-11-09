import os
import pandas as pd
from prediction_utils.extraction_utils.database import BQDatabase

from sepsis_labeler.sofa import SOFAScore 
from sepsis_labeler.cohort import SepsisAdmissionCohort
from sepsis_labeler.component import * 
from sepsis_labeler.starr_flowsheet_extract import STARRFlowsheetExtract 

class SepsisLabeler:
	
	def __init__(self, *args, **kwargs):
		self.config_dict = self.get_config_dict(**kwargs)
		self.db = BQDatabase(**self.config_dict)
		self.verbose = self.config_dict['verbose']
	
	def create_labels(self):
		if self.verbose:
			print('Running SEPSIS labeler...')
		# Extract flowsheet values into their own table if applicable
		self.extract_flowsheets()
		
		# Create cohort if pre-existing cohort is not defined
		self.create_cohort()
		
		# Create component tables
		self.create_components()
		
		# Generate SOFA score differences
		self.create_sofa()
		
		# create labelled cohort
		df =  self.create_labelled_cohort()
		
		if self.verbose:
			print('Finished!')	
		if df is not None:
			print('Returning labels as dataframe...')
			return df
	
	def extract_flowsheets(self):
		if self.config_dict['extract_flowsheet']:
			if self.verbose:
				print('Extracting flowsheets from observation table...')
			STARRFlowsheetExtract(**self.config_dict).create_cohort_table()
	
	def create_cohort(self):
		if self.config_dict['pre_existing_cohort']:
			if self.verbose:
				print(f'Using pre-existing admission cohort: {self.config_dict["cohort_name"]}')
		else:
			if self.verbose:
				print(f'Creating admission cohort: {self.config_dict["admission_rollup"]}')
			SepsisAdmissionCohort(**self.config_dict).create_cohort_table()
			if self.verbose:
				print(f'Admission cohort created...')
	
	def create_sofa(self):
		if self.verbose:
				print(f'Creating SOFA scores...')
		SOFAScore(**self.config_dict).create_sofa_tables()
	
	def create_labelled_cohort(self):
		if self.verbose:
			print('Adding SEPSIS label to cohort...')
		if self.config_dict["replace_cohort"]:
			cohort_name = self.config_dict['admission_rollup']
		else:
			cohort_name = f"{self.config_dict['admission_rollup']}_labeled"
		query = '''
				{save_query}
				SELECT adm.*,
				sep.sepsis_index_date,
				CASE
					WHEN sep.sofa_diff = "Yes" THEN 1
					ELSE 0
				END sepsis
				FROM
				{admission_rollup} adm
				LEFT JOIN {sepsis_difference} sep on sep.person_id = adm.person_id and sep.admit_date = CAST(adm.admit_date as DATE)
				'''
		query = query.format_map({**self.config_dict,
								  **{"save_query": f"CREATE OR REPLACE TABLE `{cohort_name}` AS" if self.config_dict["save_to_database"] else ""}
								 })
		if not self.config_dict["save_to_database"]:
			return pd.read_gbq(query, dialect='standard')
		else:
			self.db.execute_sql(query)
			return None
	
	def create_components(self):
		if self.verbose:
			print('Creating suspected infection list...')
		self.get_suspected_infection()
		if self.verbose:
			print('Creating platelet component table...')
		self.get_platelet()
		if self.verbose:
			print('Creating creatinine component table...')
		self.get_creatinine()
		if self.verbose:
			print('Creating GCS component table...')
		self.get_gcs()
		if self.verbose:
			print('Creating bilirubin component table...')
		self.get_bilirubin()
		if self.verbose:
			print('Creating mechanical ventilation component table...')
		self.get_mech_vent()
		if self.verbose:
			print('Creating lactate component table...')
		self.get_lactate()
		if self.verbose:
			print('Creating PaO2-FiO2 component table...')
		self.get_pao2_fio2()
		if self.verbose:
			print('Creating SpO2-FiO2 component table...')
		self.get_spo2_fio2()
		if self.verbose:
			print('Creating mean arterial pressure component table...')
		self.get_map()
		if self.verbose:
			print('Creating urine component table...')
		self.get_urine()
		if self.verbose:
			print('Creating dopamine component table...')
		self.get_dopamine()
		if self.verbose:
			print('Creating dobutamine component table...')
		self.get_dobutamine()
		if self.verbose:
			print('Creating epinephrine component table...')
		self.get_epinephrine()
		if self.verbose:
			print('Creating norepinephrine component table...')
		self.get_norepinephrine()
	
	def get_suspected_infection(self):
		SuspectedInfectionComponent(**self.config_dict).create_component_table()
	
	def get_platelet(self):
		PlateletComponent(**self.config_dict).create_component_table()
		PlateletComponent(prior=True, **self.config_dict).create_component_table()
		
	def get_creatinine(self):
		CreatinineComponent(**self.config_dict).create_component_table()
		CreatinineComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_gcs(self):
		GlasgowComaScaleComponent(**self.config_dict).create_component_table()
		GlasgowComaScaleComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_bilirubin(self):
		BilirubinComponent(**self.config_dict).create_component_table()
		BilirubinComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_mech_vent(self):
		MechanicalVentilationComponent(**self.config_dict).create_component_table()
		MechanicalVentilationComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_lactate(self):
		LactateComponent(**self.config_dict).create_component_table()
		LactateComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_pao2_fio2(self):
		PaO2FiO2Component(**self.config_dict).create_component_table()
		PaO2FiO2Component(prior=True, **self.config_dict).create_component_table()
	
	def get_spo2_fio2(self):
		SpO2FiO2Component(**self.config_dict).create_component_table()
		SpO2FiO2Component(prior=True, **self.config_dict).create_component_table()
	
	def get_map(self):
		MeanArterialPressureComponent(**self.config_dict).create_component_table()
		MeanArterialPressureComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_urine(self):
		UrineComponent(**self.config_dict).create_component_table()
		UrineComponent(prior=True, **self.config_dict).create_component_table()
	
	def get_dopamine(self):
		DopamineComponent(**self.config_dict).create_component_table()
		DopamineComponent(prior=True, **self.config_dict).create_component_table()

	def get_dobutamine(self):
		DobutamineComponent(**self.config_dict).create_component_table()
		DobutamineComponent(prior=True, **self.config_dict).create_component_table()

	def get_epinephrine(self):
		EpinephrineComponent(**self.config_dict).create_component_table()
		EpinephrineComponent(prior=True, **self.config_dict).create_component_table()

	def get_norepinephrine(self):
		NorepinephrineComponent(**self.config_dict).create_component_table()
		NorepinephrineComponent(prior=True, **self.config_dict).create_component_table()
		
	def get_defaults(self):
		
		config_dict = {
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
			"verbose":True,
			"print_query":False,
			"pre_existing_cohort":None,
			"extract_flowsheet":False,
			"save_to_database":True,
			"replace_cohort":True,
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
		
		return config_dict

	def override_defaults(self, **kwargs):
		return {**self.get_defaults(), **kwargs}

	def get_config_dict(self, **kwargs):
		config_dict = self.override_defaults(**kwargs)
		
		table_names = {
			"admission_rollup": config_dict['cohort_name'] if config_dict['pre_existing_cohort'] else "sepsis_admission_rollup",
			"suspected_infection": "sepsis_susp_inf_rollup",
			"sepsis_platelet": "sepsis_platelet_rollup",
			"sepsis_creatinine": "sepsis_creatinine_rollup",
			"sepsis_bilirubin": "sepsis_bilirubin_rollup",
			"sepsis_lactate": "sepsis_lactate_rollup",
			"sepsis_urine": "sepsis_urine_rollup",
			"sepsis_pao2_fio2": "sepsis_pao2_fio2_rollup",
			"sepsis_spo2_fio2": "sepsis_spo2_fio2_rollup",
			"sepsis_map": "sepsis_map_rollup",
			"sepsis_gcs": "sepsis_gcs_rollup",
			"sepsis_vent": "sepsis_vent_rollup",
			"sepsis_dopamine": "sepsis_dopamine_rollup",
			"sepsis_dobutamine": "sepsis_dobutamine_rollup",
			"sepsis_epinephrine": "sepsis_epinephrine_rollup",
			"sepsis_norepinephrine": "sepsis_norepinephrine_rollup",
			"sepsis_sofa":"sepsis_sofa_score",
			"sepsis_difference": "sepsis_sofa_difference"
		}
		table_names_long = {
			key: "{rs_dataset_project}.{rs_dataset}.{table_name}".format(
				table_name=value, **config_dict
			)
			for key, value in table_names.items()
		}
		
		config_dict = {**config_dict, **table_names_long}

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