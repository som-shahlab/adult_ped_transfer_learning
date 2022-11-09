from prediction_utils.cohorts.cohort import BQCohort
from prediction_utils.extraction_utils.database import BQDatabase
import os

class STARRFlowsheetExtract(BQCohort):
	def __init__(self, *args, **kwargs):
		self.config_dict = self.get_config_dict(**kwargs)
		self.db = BQDatabase(**self.config_dict)

	def create_cohort_table(self):
		"""
		Extracts the unmapped flowsheet values in the observation table and stores as a table in the database
		"""
		self.db.execute_sql(self.get_extract_flowsheets_query())

	def get_extract_flowsheets_query(self, *args, **kwargs):
		query = '''
				create or replace table {dataset_project}.{rs_dataset}.{ext_flwsht_table} as 
				(
					with meas as (
						select observation_id, 
						json_extract_scalar(v, '$.source') as val_source,
						json_extract_scalar(v, '$.value') as val_value

						FROM `{dataset_project}.{dataset}.observation` ob 
						left join unnest(json_extract_array(value_as_string,'$.values')) as v
						where ob.observation_concept_id = 2000006253
						and JSON_VALUE(v,'$.source') = "ip_flwsht_meas.meas_value"
					),

					disp as (
						select observation_id, 
						json_extract_scalar(v, '$.source') as val_source,
						json_extract_scalar(v, '$.value') as val_value

						FROM `{dataset_project}.{dataset}.observation` ob 
						left join unnest(json_extract_array(value_as_string,'$.values')) as v
						where ob.observation_concept_id = 2000006253
						and JSON_VALUE(v,'$.source') = "ip_flo_gp_data.disp_name"
					),

					unit as (
						select observation_id, 
						json_extract_scalar(v, '$.source') as val_source,
						json_extract_scalar(v, '$.value') as val_value

						FROM `{dataset_project}.{dataset}.observation` ob 
						left join unnest(json_extract_array(value_as_string,'$.values')) as v
						where ob.observation_concept_id = 2000006253
						and JSON_VALUE(v,'$.source') = "ip_flo_gp_data.units"
					),

					src as (
						select observation_id, 
						json_extract_scalar(v, '$.source') as val_source,
						json_extract_scalar(v, '$.value') as val_value

						FROM `{dataset_project}.{dataset}.observation` ob 
						left join unnest(json_extract_array(observation_source_value,'$.values')) as v
						where ob.observation_concept_id = 2000006253
						and JSON_VALUE(v,'$.source') = "ip_flt_data.display_name"
					)

					select ob.observation_id, ob.person_id, vo.visit_occurrence_id, ob.observation_datetime,
					case 
						when ob.observation_concept_id = 2000006253
							then src.val_value 
						else ob.observation_source_value
					END as source_display_name,
					case 
						when ob.observation_concept_id = 2000006253
							then disp.val_value
						else cpt.concept_name
					END as display_name,
					case 
						when ob.observation_concept_id = 2000006253
							then meas.val_value
						when ob.observation_concept_id <> 2000006253 and value_as_string is not null
							then value_as_string
						when ob.observation_concept_id <> 2000006253 and value_as_string is null
							then CAST(value_as_number as string)
					END as meas_value,
					case 
						when ob.observation_concept_id = 2000006253
							then unit.val_value
						else ob.unit_source_value
					END as units,
					from {dataset_project}.{dataset}.observation` ob 
					left join meas on ob.observation_id = meas.observation_id
					left join unit on ob.observation_id = unit.observation_id 
					left join disp on ob.observation_id = disp.observation_id
					left join src on ob.observation_id = src.observation_id
					left join `{dataset_project}.{dataset}.concept` cpt on cpt.concept_id = ob.observation_source_concept_id
					left join `{dataset_project}.{dataset}.visit_occurrence` vo on ob.person_id = vo.person_id 
					and ob.observation_datetime >= vo.visit_start_DATETIME and ob.observation_datetime <= vo.visit_end_DATETIME
				)
		'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_defaults(self):
		return {
			"gcloud_project": "som-nero-nigam-starr",
			"dataset_project": None,
			"rs_dataset_project": None,
			"dataset": "starr_omop_cdm5_deid_2022_08_01",
			"rs_dataset": "sepsis_temp_dataset",
			"cohort_name": "sepsis_temp_cohort",
			"ext_flwsht_table":"meas_vals_json",
			"google_application_credentials": os.path.expanduser(
				"~/.config/gcloud/application_default_credentials.json"
			),
			"limit": None,
			"min_stay_hour":0,
			"print_query":False,
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