import os
from prediction_utils.extraction_utils.database import BQDatabase
from sepsis_labeler.component_base import Component

class SuspectedInfectionComponent(Component):
	'''
	Class to get suspected infections in cohort.
	'''
	def __init__(self, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{suspected_infection}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from susp_inf_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map({**self.config_dict,**{"values_query":self.get_values_query(), "window_query":self.get_window_query(), "rollup_query":self.get_rollup_query()}})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH 
				blood_culture_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id = 4107893
				),
				blood_culture_from_measurement_via_ancestor AS (  
					SELECT 
						person_id, 
						measurement_DATETIME
					FROM {dataset_project}.{dataset}.measurement AS measure
					WHERE measure.measurement_concept_id IN (
						SELECT concept_id
						FROM blood_culture_list)
				),
				systemic_abx_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id = 21602796 
				),
				systemic_abx_from_drug_exposure_via_ancestor AS (
					SELECT 
						person_id, 
						drug_concept_id, 
						drug_exposure_start_DATETIME
					FROM {dataset_project}.{dataset}.drug_exposure AS drug
					WHERE drug.drug_concept_id IN (
						SELECT concept_id
						FROM systemic_abx_list)
				),
				systemic_abx_from_drug_exposure_with_name AS (
					SELECT 
						systemic_abx.*, 
						concept.concept_name AS systemic_abx_type  
					FROM systemic_abx_from_drug_exposure_via_ancestor AS systemic_abx
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON systemic_abx.drug_concept_id = concept.concept_id
				),
				bc_abx AS (
					SELECT 
						blood_culture.person_id, 
						blood_culture.measurement_DATETIME as bc_DATETIME,
						systemic_abx.drug_exposure_start_DATETIME, 
						systemic_abx.systemic_abx_type
					FROM blood_culture_from_measurement_via_ancestor AS blood_culture
					LEFT JOIN systemic_abx_from_drug_exposure_with_name AS systemic_abx
					ON blood_culture.person_id = systemic_abx.person_id
				),
				admit_bc_abx AS (
					SELECT 
						admission_rollup.*, 
						bc_abx.bc_DATETIME, 
						bc_abx.drug_exposure_start_DATETIME, 
						bc_abx.systemic_abx_type   
					FROM `{admission_rollup}` as admission_rollup
					LEFT JOIN bc_abx AS bc_abx
					ON admission_rollup.person_id = bc_abx.person_id
				),
				admit_age AS (
					SELECT
						admit_bc_abx.*,
						DATETIME_DIFF(admit_bc_abx.admit_date, p.birth_datetime, MONTH) as age_in_months
						FROM admit_bc_abx
						LEFT JOIN {dataset_project}.{dataset}.person p on admit_bc_abx.person_id = p.person_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				susp_inf_window AS (
					SELECT 
						person_id, 
						bc_DATETIME, 
						drug_exposure_start_DATETIME,
						age_in_months,
						admit_date as admit_datetime,
						cast(admit_date as DATE) as admit_date,
						discharge_date as discharge_datetime,
						cast(discharge_date as DATE) as discharge_date, 
						systemic_abx_type,
						datetime_diff(drug_exposure_start_DATETIME, bc_DATETIME, DAY) as days_bc_abx
					FROM admit_age as admit_age
					WHERE
						CAST(bc_DATETIME AS DATE) >= DATE_SUB(admit_date, INTERVAL 1 DAY) AND  CAST(bc_DATETIME AS DATE) <= discharge_date AND
						CAST(drug_exposure_start_DATETIME AS DATE) >= DATE_SUB(admit_date, INTERVAL 1 DAY) AND  CAST(drug_exposure_start_DATETIME AS DATE) <= discharge_date
					AND
						CAST(bc_DATETIME AS DATE)<= CAST(DATETIME_ADD(drug_exposure_start_DATETIME, INTERVAL 1 DAY) AS DATE) AND
						CAST(bc_DATETIME AS DATE)>= CAST(DATETIME_SUB(drug_exposure_start_DATETIME, INTERVAL 3 DAY) AS DATE) 
					ORDER BY person_id, admit_date, bc_DATETIME, drug_exposure_start_DATETIME, age_in_months
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_rollup_query(self, format_query=True):
		query = '''
				susp_inf_rollup AS (
					SELECT 
					person_id, 
					age_in_months,
					admit_date,
					admit_datetime,
					MIN(discharge_date) AS discharge_date,
					MIN(discharge_datetime) as discharge_datetime,
					MIN(bc_DATETIME) as min_bc,
					MIN(drug_exposure_start_DATETIME) as min_systemic_abx,
					LEAST(MIN(bc_DATETIME),MIN(drug_exposure_start_DATETIME)) as sepsis_index_date
				FROM susp_inf_window 
				GROUP BY person_id, admit_date, admit_datetime, age_in_months
				ORDER BY person_id, admit_date, admit_datetime, age_in_months
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class PlateletComponent(Component): 
	'''
	Class to get platelet count for cohort.
	Units are 1000/uL
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_platelet}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				SELECT * FROM platelet_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_platelet":self.config_dict['sepsis_platelet'] + '_prior' if self.prior else self.config_dict['sepsis_platelet']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH platelet_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						measure.value_as_number, 
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id in(
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						WHERE c.concept_id in (37037425,40654106)
						UNION DISTINCT
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						INNER JOIN {dataset_project}.{dataset}.concept_ancestor ca
						ON c.concept_id = ca.descendant_concept_id
						AND ca.ancestor_concept_id in (37037425,40654106)
						AND c.invalid_reason is null
					) AND measure.value_as_number IS NOT NULL AND measure.value_as_number > 0
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				platelet_window AS (
					SELECT 
						susp_inf_rollup.*, 
						platelet.measurement_DATETIME AS platelet_date, 
						platelet.value_as_number,
						datetime_diff(platelet.measurement_DATETIME, sepsis_index_date, DAY) as days_plat_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN platelet_from_measurement AS platelet USING (person_id)
					WHERE
						{window}
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				platelet_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MIN(value_as_number) as min_platelet
					FROM platelet_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class CreatinineComponent(Component): 
	'''
	Class to get creatinine measurement for cohort.
	Units are normalized to mg/dL
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_creatinine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from creatinine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_creatinine":self.config_dict['sepsis_creatinine'] + '_prior' if self.prior else self.config_dict['sepsis_creatinine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH creatinine_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						case
							when unit_concept_id = 8749 then value_as_number / 0.0113122 -- umol/l -> mg/dL
							when unit_concept_id = 8837 then value_as_number / 1000 -- ug/dL -> mg/dL
							else value_as_number 
						end as value_as_number, 
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id in (
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						WHERE c.concept_id in (37029387,4013964,2212294,3051825)
						UNION DISTINCT
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						INNER JOIN {dataset_project}.{dataset}.concept_ancestor ca
						ON c.concept_id = ca.descendant_concept_id
						AND ca.ancestor_concept_id in (37029387,4013964,2212294,3051825)
						AND c.invalid_reason is null
					) AND measure.value_as_number IS NOT NULL AND measure.value_as_number > 0
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				creatinine_window AS (
					SELECT 
						susp_inf_rollup.*, 
						creatinine.measurement_DATETIME AS creatinine_date, 
						creatinine.value_as_number,
						datetime_diff(creatinine.measurement_DATETIME, sepsis_index_date, DAY) as days_crea_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN creatinine_from_measurement AS creatinine USING (person_id)
					WHERE
						{window}
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				creatinine_rollup AS (
					SELECT 
						person_id, 
						admit_date,
						MAX(value_as_number) as max_creatinine
					FROM creatinine_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class GlasgowComaScaleComponent(Component): 
	'''
	Class to get glasgow coma scale score measurement for cohort.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_gcs}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from gcs_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_gcs":self.config_dict['sepsis_gcs'] + '_prior' if self.prior else self.config_dict['sepsis_gcs']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH gcs_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						measure.value_as_number, 
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id = 3032652 AND measure.value_as_number >= 3 AND measure.value_as_number <= 15
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				gcs_window AS (
					SELECT 
						susp_inf_rollup.*, 
						gcs.measurement_DATETIME AS gcs_date, 
						gcs.value_as_number,
						datetime_diff(gcs.measurement_DATETIME, sepsis_index_date, DAY) as days_gcs_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN gcs_from_measurement AS gcs USING (person_id)
					WHERE
						{window}
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				gcs_rollup AS (
					SELECT 
						person_id, 
						admit_date,
						MIN(value_as_number) as min_gcs
					FROM gcs_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class BilirubinComponent(Component): 
	'''
	Class to get bilirubin measurement for cohort.
	Units are normalized to mg/dL
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_bilirubin}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from bilirubin_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_bilirubin":self.config_dict['sepsis_bilirubin'] + '_prior' if self.prior else self.config_dict['sepsis_bilirubin']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH bilirubin_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						value_as_number, 
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id in (
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						WHERE c.concept_id in (3024128, 4230543)
						UNION DISTINCT
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						INNER JOIN {dataset_project}.{dataset}.concept_ancestor ca
						ON c.concept_id = ca.descendant_concept_id
						AND ca.ancestor_concept_id in (3024128, 4230543)
						AND c.invalid_reason is null
					) AND measure.value_as_number IS NOT NULL AND measure.value_as_number > 0
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				bilirubin_window AS (
					SELECT 
						susp_inf_rollup.*, 
						bilirubin.measurement_DATETIME AS bilirubin_date, 
						bilirubin.value_as_number,
						datetime_diff(bilirubin.measurement_DATETIME, sepsis_index_date, DAY) as days_bili_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN bilirubin_from_measurement AS bilirubin USING (person_id)
					WHERE
						{window}
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				bilirubin_rollup AS (
					SELECT 
						person_id, 
						admit_date,
						MAX(value_as_number) as max_bilirubin
					FROM bilirubin_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)
		
class MechanicalVentilationComponent(Component): 
	'''
	Class to get mechanical ventilation measurement for cohort.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_vent}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from mech_vent_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_vent":self.config_dict['sepsis_vent'] + '_prior' if self.prior else self.config_dict['sepsis_vent']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH mech_vent_from_flowsheet AS (
					SELECT 
						person_id, 
						observation_datetime, 
						UPPER(display_name) as row_disp_name, 
						meas_value, 
						unit_value as units
					FROM {dataset_project}.{rs_dataset}.{ext_flwsht_table}
					where (upper(display_name) = 'VENT MODE' or upper(display_name) = 'VENTILATION MODE') and
					(meas_value <> 'STANDBY' and meas_value <> 'MONITOR' and meas_value is not null)   
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				mech_vent_window AS (
					SELECT 
						susp_inf_rollup.*, 
						mech_vent.observation_datetime AS mech_vent_datetime, 
						mech_vent.meas_value as vent_mode,
						datetime_diff(mech_vent.observation_datetime, sepsis_index_date, DAY) as days_mech_vent_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN mech_vent_from_flowsheet AS mech_vent USING (person_id)
					WHERE
						{window}
					ORDER BY  person_id, admit_date, sepsis_index_date, observation_datetime 
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['obs_window_prior'] if self.prior else self.config_dict['obs_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				mech_vent_rollup AS (
					SELECT 
						person_id, 
						admit_date,
						COUNT(vent_mode) as count_vent_mode
					FROM mech_vent_window
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class LactateComponent(Component): 
	'''
	Class to get lactate measurement for cohort.
	Units are normalized to mmol/L
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_lactate}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from lactate_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_lactate":self.config_dict['sepsis_lactate'] + '_prior' if self.prior else self.config_dict['sepsis_lactate']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH lactate_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME,
						case
							when unit_concept_id = 8840 then value_as_number * 9.0 -- mg/dL -> mmol/L
							when unit_concept_id = 8837 then value_as_number / 0.001 * 9.0 -- ug/dL -> mmol/L
							else value_as_number 
						end as value_as_number,
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id in(
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						WHERE c.concept_id in (3047181, 40762125, 3014111, 3020138)
						UNION DISTINCT
						SELECT
							c.concept_id
						FROM {dataset_project}.{dataset}.concept c
						INNER JOIN {dataset_project}.{dataset}.concept_ancestor ca
						ON c.concept_id = ca.descendant_concept_id
						AND ca.ancestor_concept_id in (3047181, 40762125, 3014111, 3020138)
						AND c.invalid_reason is null
					) AND measure.value_as_number IS NOT NULL AND measure.value_as_number > 0
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				lactate_window AS (
					SELECT 
						susp_inf_rollup.*, 
						lactate.measurement_DATETIME AS lactate_date, 
						lactate.value_as_number,
						datetime_diff(lactate.measurement_DATETIME, sepsis_index_date, DAY) as days_lact_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN lactate_from_measurement AS lactate USING (person_id)
					WHERE
						{window}
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				lactate_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MAX(value_as_number) as max_lactate
					FROM lactate_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)
		
class PaO2FiO2Component(Component): 
	'''
	Class to get PaO2:FiO2 ratio measurement for cohort.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_pao2_fio2}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from paO2_fiO2_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_pao2_fio2":self.config_dict['sepsis_pao2_fio2'] + '_prior' if self.prior else self.config_dict['sepsis_pao2_fio2']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH paO2_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						measure.value_as_number, 
						concept.concept_name AS measure_type  
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					WHERE concept.concept_id=3027801 AND value_as_number IS NOT NULL
				),
				fiO2_vals as (
					SELECT 
						person_id, 
						observation_datetime, 
						UPPER(display_name) as row_disp_name, 
						SAFE_CAST(meas_value as float64) as meas_value, 
						unit_value as units
					FROM {dataset_project}.{rs_dataset}.{ext_flwsht_table}
					where (upper(display_name) = 'FIO2 (%)' or (upper(display_name) = 'FIO2 %' and upper(source_display_name) = 'RN CLINICAL SCREENING'))
						  and meas_value is not null
				),
				fiO2_from_flowsheet AS (
					select 
						person_id, 
						observation_datetime, 
						row_disp_name,
						case 
							when meas_value > 1 then meas_value / 100
							else meas_value
						end as fiO2,
						units
					from fiO2_vals
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				paO2_window AS (
					SELECT 
						susp_inf_rollup.*, 
						paO2.measurement_DATETIME AS paO2_datetime, 
						paO2.value_as_number as paO2,
						datetime_diff(paO2.measurement_DATETIME, sepsis_index_date, DAY) as days_paO2_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN paO2_from_measurement AS paO2
					USING (person_id)
					WHERE
						{meas_window}
				),
				fiO2_window AS ( 
					SELECT 
						susp_inf_rollup.person_id, 
						min_bc, 
						min_systemic_abx, 
						susp_inf_rollup.admit_date, 
						sepsis_index_date, 
						CAST(observation_datetime AS DATETIME) AS fiO2_datetime, 
						fiO2, 
						datetime_diff(CAST(observation_datetime AS DATETIME), sepsis_index_date, DAY) as days_fiO2_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN fiO2_from_flowsheet as flowsheet
					ON susp_inf_rollup.person_id = flowsheet.person_id
					WHERE
					{obs_window} AND
					fiO2 >=0.21 AND fiO2 <=1.0
				),
				paO2_fiO2_window AS (
					SELECT 
						paO2_window.person_id, 
						paO2_window.admit_date, 
						paO2_window.sepsis_index_date, 
						fiO2, 
						fiO2_datetime, 
						paO2, 
						paO2_datetime,  
						paO2/(NULLIF(fiO2, 0)) AS paO2fiO2_ratio, 
						datetime_diff(paO2_datetime, fiO2_datetime, MINUTE) as minutes_fiO2_paO2
					FROM fiO2_window AS fiO2_window
					INNER JOIN paO2_window AS paO2_window
					USING (person_id, sepsis_index_date)
					WHERE CAST(fiO2_datetime AS DATETIME)<= paO2_datetime 
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(
					{
						**self.config_dict,
						**{"meas_window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window'],
						   "obs_window":self.config_dict['obs_window_prior'] if self.prior else self.config_dict['obs_window']}
					})

	def get_rollup_query(self, format_query=True):
		query = '''
				paO2_fiO2_initial_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						paO2_datetime, 
						MIN(minutes_fiO2_paO2) As minutes_fiO2_paO2 
					FROM paO2_fiO2_window 
					GROUP BY person_id, admit_date, paO2_datetime
				),
				paO2_fiO2_initial_rollup_join AS (
					SELECT 
						initial_rollup.person_id, 
						initial_rollup.admit_date, 
						initial_rollup.paO2_datetime, 
						initial_rollup.minutes_fiO2_paO2,
						sepsis_index_date, 
						fiO2, fiO2_datetime, paO2, paO2fiO2_ratio
					FROM paO2_fiO2_initial_rollup AS initial_rollup
					LEFT JOIN paO2_fiO2_window AS combined_window
					USING (person_id, paO2_datetime, minutes_fiO2_paO2)
				),
				paO2_fiO2_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MIN(paO2fiO2_ratio) as min_paO2fiO2_ratio
					FROM paO2_fiO2_initial_rollup_join 
					WHERE minutes_fiO2_paO2 <= 24*60
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class SpO2FiO2Component(Component): 
	'''
	Class to get SpO2:FiO2 measurement ratio for cohort.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_spo2_fio2}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from spO2_fiO2_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_spo2_fio2":self.config_dict['sepsis_spo2_fio2'] + '_prior' if self.prior else self.config_dict['sepsis_spo2_fio2']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH spO2_from_flowsheet AS (
					SELECT 
						person_id, 
						observation_datetime, 
						UPPER(display_name) as row_disp_name, 
						SAFE_CAST(meas_value as float64) as meas_value, 
						unit_value as units
					FROM {dataset_project}.{rs_dataset}.{ext_flwsht_table}
					where ((upper(display_name) = 'OXYGEN SATURATION' AND upper(source_display_name) = 'SPO2') 
					or (upper(display_name) like 'SPO2 - %' and upper(source_display_name) = 'DEVICES TESTING TEMPLATE'))
					and meas_value is not null
				),
				fiO2_vals as (
					SELECT 
						person_id, 
						observation_datetime, 
						UPPER(display_name) as row_disp_name, 
						SAFE_CAST(meas_value as float64) as meas_value, 
						unit_value as units
					FROM {dataset_project}.{rs_dataset}.{ext_flwsht_table}
					where (upper(display_name) = 'FIO2 (%)' or (upper(display_name) = 'FIO2 %' and upper(source_display_name) = 'RN CLINICAL SCREENING'))
						  and meas_value is not null
				),
				fiO2_from_flowsheet AS (
					select 
						person_id, 
						observation_datetime, 
						row_disp_name,
						case 
							when meas_value > 1 then meas_value / 100
							else meas_value
						end as fiO2,
						units
					from fiO2_vals
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				spO2_window AS ( 
					SELECT 
						susp_inf_rollup.person_id, 
						min_bc, 
						min_systemic_abx, 
						susp_inf_rollup.admit_date, 
						sepsis_index_date, 
						CAST(observation_datetime AS DATETIME) AS spO2_datetime, 
						meas_value as spO2, 
						datetime_diff(CAST(observation_datetime AS DATETIME), sepsis_index_date, DAY) as days_spO2_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN spO2_from_flowsheet as flowsheet
					ON susp_inf_rollup.person_id = flowsheet.person_id
					WHERE
					{window} AND
					meas_value >0 AND meas_value <=100
				),
				fiO2_window AS ( 
					SELECT 
						susp_inf_rollup.person_id, 
						min_bc, 
						min_systemic_abx, 
						susp_inf_rollup.admit_date, 
						sepsis_index_date, 
						CAST(observation_datetime AS DATETIME) AS fiO2_datetime, 
						fiO2, 
						datetime_diff(CAST(observation_datetime AS DATETIME), sepsis_index_date, DAY) as days_fiO2_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN fiO2_from_flowsheet as flowsheet
					ON susp_inf_rollup.person_id = flowsheet.person_id
					WHERE
					{window} AND
					fiO2 >=0.21 AND fiO2 <=1.0
				),
				spO2_fiO2_window AS (
					SELECT 
						spO2_window.person_id, 
						spO2_window.admit_date, 
						spO2_window.sepsis_index_date, 
						fiO2, 
						fiO2_datetime, 
						spO2, 
						spO2_datetime,  
						spO2/(NULLIF(fiO2, 0)) AS spO2fiO2_ratio, 
						datetime_diff(spO2_datetime, fiO2_datetime, MINUTE) as minutes_fiO2_spO2
					FROM fiO2_window AS fiO2_window
					INNER JOIN spO2_window AS spO2_window
					USING (person_id, sepsis_index_date)
					WHERE CAST(fiO2_datetime AS DATETIME)<= spO2_datetime 
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['obs_window_prior'] if self.prior else self.config_dict['obs_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				spO2_fiO2_initial_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						spO2_datetime, 
						MIN(minutes_fiO2_spO2) As minutes_fiO2_spO2 
					FROM spO2_fiO2_window 
					GROUP BY person_id, admit_date, spO2_datetime
				),
				spO2_fiO2_initial_rollup_join AS (
					SELECT 
						initial_rollup.person_id, 
						initial_rollup.admit_date, 
						initial_rollup.spO2_datetime, 
						initial_rollup.minutes_fiO2_spO2,
						sepsis_index_date, 
						fiO2, 
						fiO2_datetime, 
						spO2, 
						spO2fiO2_ratio
					FROM spO2_fiO2_initial_rollup AS initial_rollup
					LEFT JOIN spO2_fiO2_window AS combined_window
					USING (person_id, spO2_datetime, minutes_fiO2_spO2)
				),
				spO2_fiO2_rollup AS (
					SELECT
						person_id, 
						admit_date as admit_date, 
						MIN(spO2fiO2_ratio) as min_spO2fiO2_ratio
					FROM spO2_fiO2_initial_rollup_join 
					WHERE minutes_fiO2_spO2 <= 24*60
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)
		
class DopamineComponent(Component): 
	'''
	Class to get dopamine drug exposure for cohort.
	Dosage amounts are unavailable in STARR-OMOP, so only the presence of the drug exposure will be used.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_dopamine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from dopamine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_dopamine":self.config_dict['sepsis_dopamine'] + '_prior' if self.prior else self.config_dict['sepsis_dopamine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH dopamine_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id IN (21600284)
				),
				dopamine_from_drug_exposure_via_ancestor AS ( 
					SELECT *
					FROM {dataset_project}.{dataset}.drug_exposure AS drug
					WHERE drug.drug_concept_id IN (
						SELECT concept_id
						FROM dopamine_list
					)
				),
				dopamine_from_drug_exposure_with_name AS (
					SELECT 
						dopamine.*, 
						concept.concept_name AS dopamine_type  
					FROM dopamine_from_drug_exposure_via_ancestor AS dopamine
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON dopamine.drug_concept_id = concept.concept_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				dopamine_window AS (
					SELECT 
						susp_inf_rollup.person_id, 
						susp_inf_rollup.admit_date, 
						susp_inf_rollup.sepsis_index_date,
						dopamine.drug_exposure_start_DATETIME, 
						dopamine.drug_exposure_end_DATETIME,
						datetime_diff(dopamine.drug_exposure_start_DATETIME, sepsis_index_date, DAY) as days_index_dopamine_start,
						datetime_diff(dopamine.drug_exposure_end_DATETIME, sepsis_index_date, DAY) as days_index_dopamine_end,
						(datetime_diff(dopamine.drug_exposure_end_DATETIME, dopamine.drug_exposure_start_DATETIME, DAY) + 1) as days_dopamine
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN dopamine_from_drug_exposure_with_name AS dopamine
					USING (person_id)
					WHERE
						{window}
				),
				'''
		
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['drug_window_prior'] if self.prior else self.config_dict['drug_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				dopamine_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MAX(datetime_diff(dopamine.drug_exposure_end_DATETIME, dopamine.drug_exposure_start_DATETIME, DAY) + 1)
					as max_dopamine_days
					FROM dopamine_window as dopamine 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class DobutamineComponent(Component): 
	'''
	Class to get dobutamine drug exposure for cohort.
	Dosage amounts are unavailable in STARR-OMOP, so only the presence of the drug exposure will be used.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_dobutamine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from dobutamine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_dobutamine":self.config_dict['sepsis_dobutamine'] + '_prior' if self.prior else self.config_dict['sepsis_dobutamine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH dobutamine_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id IN (21600287)
				),
				dobutamine_from_drug_exposure_via_ancestor AS ( 
					SELECT *
					FROM {dataset_project}.{dataset}.drug_exposure AS drug
					WHERE drug.drug_concept_id IN (
						SELECT concept_id
						FROM dobutamine_list
					)
				),
				dobutamine_from_drug_exposure_with_name AS (
					SELECT 
						dobutamine.*, 
						concept.concept_name AS dobutamine_type  
					FROM dobutamine_from_drug_exposure_via_ancestor AS dobutamine
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON dobutamine.drug_concept_id = concept.concept_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				dobutamine_window AS (
					SELECT 
						susp_inf_rollup.person_id, 
						susp_inf_rollup.admit_date, 
						susp_inf_rollup.sepsis_index_date,
						dobutamine.drug_exposure_start_DATETIME, 
						dobutamine.drug_exposure_end_DATETIME,
						datetime_diff(dobutamine.drug_exposure_start_DATETIME, sepsis_index_date, DAY) as days_index_dobutamine_start,
						datetime_diff(dobutamine.drug_exposure_end_DATETIME, sepsis_index_date, DAY) as days_index_dobutamine_end,
						(datetime_diff(dobutamine.drug_exposure_end_DATETIME, dobutamine.drug_exposure_start_DATETIME, DAY) + 1) as days_dobutamine
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN dobutamine_from_drug_exposure_with_name AS dobutamine
					USING (person_id)
					WHERE
						{window}
				),
				'''
		
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['drug_window_prior'] if self.prior else self.config_dict['drug_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				dobutamine_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MAX(datetime_diff(dobutamine.drug_exposure_end_DATETIME, dobutamine.drug_exposure_start_DATETIME, DAY) + 1)
					as max_dobutamine_days
					FROM dobutamine_window as dobutamine 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class EpinephrineComponent(Component): 
	'''
	Class to get epinephrine drug exposure for cohort.
	Dosage amounts are unavailable in STARR-OMOP, so only the presence of the drug exposure will be used.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_epinephrine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from epinephrine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_epinephrine":self.config_dict['sepsis_epinephrine'] + '_prior' if self.prior else self.config_dict['sepsis_epinephrine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH epinephrine_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id IN (21600303)
				),
				epinephrine_from_drug_exposure_via_ancestor AS ( 
					SELECT *
					FROM {dataset_project}.{dataset}.drug_exposure AS drug
					WHERE drug.drug_concept_id IN (
						SELECT concept_id
						FROM epinephrine_list
					)
				),
				epinephrine_from_drug_exposure_with_name AS (
					SELECT 
						epinephrine.*, 
						concept.concept_name AS epinephrine_type  
					FROM epinephrine_from_drug_exposure_via_ancestor AS epinephrine
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON epinephrine.drug_concept_id = concept.concept_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				epinephrine_window AS (
					SELECT 
						susp_inf_rollup.person_id, 
						susp_inf_rollup.admit_date, 
						susp_inf_rollup.sepsis_index_date,
						epinephrine.drug_exposure_start_DATETIME, 
						epinephrine.drug_exposure_end_DATETIME,
						datetime_diff(epinephrine.drug_exposure_start_DATETIME, sepsis_index_date, DAY) as days_index_epinephrine_start,
						datetime_diff(epinephrine.drug_exposure_end_DATETIME, sepsis_index_date, DAY) as days_index_epinephrine_end,
						(datetime_diff(epinephrine.drug_exposure_end_DATETIME, epinephrine.drug_exposure_start_DATETIME, DAY) + 1) as days_epinephrine
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN epinephrine_from_drug_exposure_with_name AS epinephrine
					USING (person_id)
					WHERE
						{window}
				),
				'''
		
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['drug_window_prior'] if self.prior else self.config_dict['drug_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				epinephrine_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MAX(datetime_diff(epinephrine.drug_exposure_end_DATETIME, epinephrine.drug_exposure_start_DATETIME, DAY) + 1)
					as max_epinephrine_days
					FROM epinephrine_window as epinephrine 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class NorepinephrineComponent(Component): 
	'''
	Class to get norepinephrine drug exposure for cohort.
	Dosage amounts are unavailable in STARR-OMOP, so only the presence of the drug exposure will be used.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_norepinephrine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from norepinephrine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_norepinephrine":self.config_dict['sepsis_norepinephrine'] + '_prior' if self.prior else self.config_dict['sepsis_norepinephrine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH norepinephrine_list AS (
					SELECT 
						descendant_concept_id AS concept_id
					FROM {dataset_project}.{dataset}.concept_ancestor
					WHERE ancestor_concept_id IN (21600283)
				),
				norepinephrine_from_drug_exposure_via_ancestor AS ( 
					SELECT *
					FROM {dataset_project}.{dataset}.drug_exposure AS drug
					WHERE drug.drug_concept_id IN (
						SELECT concept_id
						FROM norepinephrine_list
					)
				),
				norepinephrine_from_drug_exposure_with_name AS (
					SELECT 
						norepinephrine.*, 
						concept.concept_name AS norepinephrine_type  
					FROM norepinephrine_from_drug_exposure_via_ancestor AS norepinephrine
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON norepinephrine.drug_concept_id = concept.concept_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				norepinephrine_window AS (
					SELECT 
						susp_inf_rollup.person_id, 
						susp_inf_rollup.admit_date, 
						susp_inf_rollup.sepsis_index_date,
						norepinephrine.drug_exposure_start_DATETIME, 
						norepinephrine.drug_exposure_end_DATETIME,
						datetime_diff(norepinephrine.drug_exposure_start_DATETIME, sepsis_index_date, DAY) as days_index_norepinephrine_start,
						datetime_diff(norepinephrine.drug_exposure_end_DATETIME, sepsis_index_date, DAY) as days_index_norepinephrine_end,
						(datetime_diff(norepinephrine.drug_exposure_end_DATETIME, norepinephrine.drug_exposure_start_DATETIME, DAY) + 1) as days_norepinephrine
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN norepinephrine_from_drug_exposure_with_name AS norepinephrine
					USING (person_id)
					WHERE
						{window}
				),
				'''
		
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['drug_window_prior'] if self.prior else self.config_dict['drug_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				norepinephrine_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MAX(datetime_diff(norepinephrine.drug_exposure_end_DATETIME, norepinephrine.drug_exposure_start_DATETIME, DAY) + 1)
					as max_norepinephrine_days
					FROM norepinephrine_window as norepinephrine 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class MeanArterialPressureComponent(Component): 
	'''
	Class to get MAP measurement for cohort.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_map}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from mean_arterial_pressure_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_map":self.config_dict['sepsis_map'] + '_prior' if self.prior else self.config_dict['sepsis_map']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH mean_arterial_pressure_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME, 
						measure.value_as_number, 
						concept.concept_name AS measure_type 
					FROM {dataset_project}.{dataset}.measurement AS measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					where measure.measurement_concept_id = 3027598 AND
						measure.value_as_number IS NOT NULL AND measure.value_as_number >= 10
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				mean_arterial_pressure_window AS (
					SELECT 
						susp_inf_rollup.*, 
						mapm.measurement_DATETIME AS map_datetime,
						mapm.value_as_number AS map,
						datetime_diff(mapm.measurement_DATETIME, sepsis_index_date, DAY) as days_map_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN mean_arterial_pressure_from_measurement as mapm
						ON susp_inf_rollup.person_id = mapm.person_id
					WHERE
						{window} 
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				mean_arterial_pressure_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						MIN(map) as min_map
					FROM mean_arterial_pressure_window 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

class UrineComponent(Component): 
	'''
	Class to get urine measurement for cohort.
	Urine must be summed over day for each person during window.
	'''
	def __init__(self, prior=False, *args, **kwargs):
		Component.__init__(self, *args, **kwargs)
		self.prior = prior
	
	def get_component_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_urine}` AS
				{values_query}
				{window_query}
				{rollup_query} 
				select * from urine_rollup
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"values_query":self.get_values_query(), 
							"window_query":self.get_window_query(), 
							"rollup_query":self.get_rollup_query()}, 
							"sepsis_urine":self.config_dict['sepsis_urine'] + '_prior' if self.prior else self.config_dict['sepsis_urine']})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_values_query(self, format_query=True):
		query = '''
				WITH urine_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME,
						measure.value_as_number, 
						concept.concept_name AS measure_type 
					FROM {dataset_project}.{dataset}.measurement measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					where measurement_concept_id = 45876241 and measure.value_as_number >= 0 and measure.value_as_number IS NOT NULL
				),
				urine_24_from_measurement AS (
					SELECT 
						measure.person_id, 
						measure.measurement_DATETIME,
						measure.value_as_number, 
						concept.concept_name AS measure_type 
					FROM {dataset_project}.{dataset}.measurement measure
					INNER JOIN {dataset_project}.{dataset}.concept AS concept
					ON measure.measurement_concept_id = concept.concept_id
					where measurement_concept_id = 3012565 and measure.value_as_number >= 0 and measure.value_as_number IS NOT NULL
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_window_query(self, format_query=True):
		query = '''
				urine_window AS (
					SELECT 
						susp_inf_rollup.*, 
						urine.measurement_DATETIME AS urine_datetime, 
						urine.value_as_number AS urine_volume,
						datetime_diff(measurement_DATETIME, sepsis_index_date, DAY) as days_urine_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN urine_from_measurement as urine
						ON susp_inf_rollup.person_id = urine.person_id 
					WHERE
						{window}
					ORDER BY person_id, admit_date, measurement_DATETIME
				),
				urine_24_window AS (
					SELECT 
						susp_inf_rollup.*, 
						urine.measurement_DATETIME AS urine_datetime, 
						urine.value_as_number AS urine_volume,
						datetime_diff(measurement_DATETIME, sepsis_index_date, DAY) as days_urine_index
					FROM {suspected_infection} AS susp_inf_rollup
					LEFT JOIN urine_24_from_measurement as urine
						ON susp_inf_rollup.person_id = urine.person_id 
					WHERE
						{window}
					ORDER BY person_id, admit_date, measurement_DATETIME
				),
				urine_admit_time AS (
					SELECT 
						person_id, 
						MIN(observation_datetime) AS ext_urine_datetime,
						EXTRACT(HOUR FROM MIN(observation_datetime)) AS hour, 
						(24-EXTRACT(HOUR FROM MIN(observation_datetime))) AS adjust_hours
					FROM urine_window AS urine
					LEFT JOIN {dataset_project}.{rs_dataset}.{ext_flwsht_table} AS flowsheets_orig  USING (person_id)
					WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
					AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
					GROUP BY person_id
				),
				urine_discharge_time AS (
					SELECT 
						person_id, 
						MAX(observation_datetime) AS ext_urine_datetime,
						EXTRACT(HOUR FROM MAX(observation_datetime)) AS adjust_hours
					FROM urine_window AS urine
					LEFT JOIN {dataset_project}.{rs_dataset}.{ext_flwsht_table} AS flowsheets_orig  USING (person_id)
					WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
					AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
					GROUP BY person_id
				),
				urine_24_admit_time AS (
					SELECT 
						person_id, 
						MIN(observation_datetime) AS ext_urine_datetime,
						EXTRACT(HOUR FROM MIN(observation_datetime)) AS hour, 
						(24-EXTRACT(HOUR FROM MIN(observation_datetime))) AS adjust_hours
					FROM urine_window AS urine
					LEFT JOIN {dataset_project}.{rs_dataset}.{ext_flwsht_table} AS flowsheets_orig  USING (person_id)
					WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
					AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
					GROUP BY person_id
				),
				urine_24_discharge_time AS (
					SELECT 
						person_id, 
						MAX(observation_datetime) AS ext_urine_datetime,
						EXTRACT(HOUR FROM MAX(observation_datetime)) AS adjust_hours
					FROM urine_window AS urine
					LEFT JOIN {dataset_project}.{rs_dataset}.{ext_flwsht_table} AS flowsheets_orig  USING (person_id)
					WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
					AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
					GROUP BY person_id
				),
				'''
		if not format_query:
			return query
		else:
			return query.format_map({**self.config_dict,**{"window":self.config_dict['meas_window_prior'] if self.prior else self.config_dict['meas_window']}})

	def get_rollup_query(self, format_query=True):
		query = '''
				urine_output_initial_rollup AS (
					(
						SELECT 
							person_id, 
							admit_date, 
							discharge_date, 
							CAST(urine_datetime AS DATE) AS urine_date, 
							SUM(urine_volume) as urine_daily_output_orig, 
							SUM(urine_volume) as urine_daily_output_adj,
							ext_urine_datetime, 
							adjust_hours
						FROM urine_window 
						LEFT JOIN urine_admit_time USING (person_id)
						WHERE CAST(urine_datetime AS DATE) <> admit_date AND CAST(urine_datetime AS DATE) <> CAST (discharge_date AS DATE)
						GROUP BY person_id, admit_date, discharge_date, CAST (urine_datetime AS DATE), ext_urine_datetime, adjust_hours 
					)
					UNION ALL
					(
						SELECT 
							person_id, 
							admit_date, 
							discharge_date, 
							CAST(urine_datetime AS DATE) AS urine_date, 
							SUM(urine_volume) as urine_daily_output_orig, 
							(SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
							ext_urine_datetime, 
							adjust_hours
						FROM urine_window 
						LEFT JOIN urine_admit_time USING (person_id)
						WHERE CAST(urine_datetime AS DATE) = admit_date
						GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
					)
					UNION ALL
					(
						SELECT 
							person_id, 
							admit_date, 
							discharge_date, 
							CAST(urine_datetime AS DATE) AS urine_date, 
							SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
							ext_urine_datetime, 
							adjust_hours
						FROM urine_window 
						LEFT JOIN urine_discharge_time USING (person_id)
						WHERE CAST(urine_datetime AS DATE) = discharge_date AND adjust_hours <> 0 
						GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
					)
					UNION ALL # THIS LAST BIT DEALS WITH DISCHARGE AT 0:00:00 HOURS
					(
						SELECT 
							person_id, 
							admit_date, 
							discharge_date, 
							CAST(urine_datetime AS DATE) AS urine_date, 
							SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24 as urine_daily_output_adj,
							ext_urine_datetime, 
							adjust_hours
						FROM urine_window 
						LEFT JOIN urine_discharge_time USING (person_id)
						WHERE CAST(urine_datetime AS DATE) = discharge_date AND adjust_hours = 0 
						GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
					)
				),
				urine_rollup AS (
					SELECT 
						person_id, 
						admit_date, 
						Min(urine_daily_output_adj) as min_urine_daily
					FROM urine_output_initial_rollup 
					GROUP BY person_id, admit_date
				)
				'''
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)