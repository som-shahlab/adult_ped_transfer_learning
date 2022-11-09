import os
from prediction_utils.extraction_utils.database import BQDatabase

class SOFAScore:
	def __init__(self, prior=False, *args, **kwargs):
		self.config_dict = self.get_config_dict(**kwargs)
		self.db = BQDatabase(**self.config_dict)
	
	def get_difference_query(self, format_query=True):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_difference}` AS
				SELECT 
					cohort.person_id, 
					cohort.admit_date, 
					cohort.discharge_date, 
					cohort.sepsis_index_date,
					c.min_platelet,
					c.plat_SOFA,
					p.min_platelet as min_platelet_prior,
					p.plat_SOFA as plat_SOFA_prior,
					c.max_bilirubin, 
					c.bili_SOFA,
					p.max_bilirubin as max_bilirubin_prior,
					p.bili_SOFA as bili_SOFA_prior,
					c.max_creatinine,
					c.crea_SOFA,
					p.max_creatinine as max_creatinine_prior,
					p.crea_SOFA as crea_SOFA_prior, 
					c.max_dopamine_days,
					c.max_dobutamine_days,
					c.max_epinephrine_days,
					c.max_norepinephrine_days,
					c.min_map,
					c.cv_SOFA,
					p.max_dopamine_days as max_dopamine_days_prior,
					p.max_dobutamine_days as max_dobutamine_days_prior,
					p.max_epinephrine_days as max_epinephrine_days_prior,
					p.max_norepinephrine_days as max_norepinephrine_days_prior,
					p.min_map as min_map_prior,
					p.cv_SOFA as cv_SOFA_prior,
					c.count_vent_mode,
					c.min_pao2fio2_ratio,
					c.pao2_resp_SOFA as pao2_resp_SOFA,
					c.min_spo2fio2_ratio,
					c.spo2_resp_SOFA as spo2_resp_SOFA,
					GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) as resp_SOFA,
					p.count_vent_mode as count_vent_mode_prior,
					p.min_pao2fio2_ratio as min_pao2fio2_ratio_prior,
					p.pao2_resp_SOFA as pao2_resp_SOFA_prior,
					p.min_spo2fio2_ratio as min_spo2fio2_ratio_prior,
					p.spo2_resp_SOFA as spo2_resp_SOFA_prior,
					GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) as resp_SOFA_prior,
					c.min_gcs,
					c.gcs_SOFA,
					p.min_gcs as min_gcs_prior,
					p.gcs_SOFA as gcs_SOFA_prior,
					c.min_urine_daily,
					c.urine_SOFA,
					p.min_urine_daily as min_urine_daily_prior,
					p.urine_SOFA as urine_SOFA_prior,
					c.max_lactate,
					p.max_lactate as max_lactate_prior,
					(c.plat_SOFA + c.bili_SOFA + c.crea_SOFA + c.cv_SOFA + GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) + c.gcs_SOFA + c.urine_SOFA) AS SOFA_score_current,
					(p.plat_SOFA + p.bili_SOFA + p.crea_SOFA + p.cv_SOFA + GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) + p.gcs_SOFA + p.urine_SOFA) AS SOFA_score_prior,  
					((c.plat_SOFA + c.bili_SOFA + c.crea_SOFA + c.cv_SOFA + GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) + c.gcs_SOFA + c.urine_SOFA) - 
					 (p.plat_SOFA + p.bili_SOFA + p.crea_SOFA + p.cv_SOFA + GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) + p.gcs_SOFA + p.urine_SOFA)) AS SOFA_score_diff,
					CASE 
						WHEN (c.plat_SOFA + c.bili_SOFA + c.crea_SOFA + c.cv_SOFA + GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) + c.gcs_SOFA + c.urine_SOFA) >=2 THEN "Yes" 
						ELSE "No" 
					END SOFA_current,
					CASE 
						WHEN (p.plat_SOFA + p.bili_SOFA + p.crea_SOFA + p.cv_SOFA + GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) + p.gcs_SOFA + p.urine_SOFA) >=2 THEN "Yes" 
						ELSE "No" 
					END SOFA_prior,
					CASE 
						WHEN ((c.plat_SOFA + c.bili_SOFA + c.crea_SOFA + c.cv_SOFA + GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) + c.gcs_SOFA + c.urine_SOFA) - 
							  (p.plat_SOFA + p.bili_SOFA + p.crea_SOFA + p.cv_SOFA + GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) + p.gcs_SOFA + p.urine_SOFA)) >=2 THEN "Yes" 
						ELSE "No" 
					END SOFA_diff,
					c.vaso_shock, 
					c.map_shock, 
					c.lact_shock, 
					(c.vaso_shock + c.map_shock + c.lact_shock) AS shock_score,
					p.vaso_shock as vaso_shock_prior, 
					p.map_shock as map_shock_prior, 
					p.lact_shock as lact_shock_prior, 
					(p.vaso_shock + p.map_shock + p.lact_shock ) as shock_score_prior,
					CASE WHEN (c.vaso_shock + c.map_shock + c.lact_shock) = 2 THEN "Yes" ELSE "No" END shock,  
					CASE WHEN (p.vaso_shock + p.map_shock + p.lact_shock) = 2 THEN "Yes" ELSE "No" END shock_prior,  
					CASE WHEN ((c.vaso_shock + c.map_shock + c.lact_shock) - (p.vaso_shock + p.map_shock + p.lact_shock)) = 2 THEN "Yes" ELSE "No" END shock_diff, 
					CASE WHEN c.plat_SOFA >= 2 THEN 1 ELSE 0 END plat_SOFA_GT2,
					CASE WHEN c.bili_SOFA >= 2 THEN 1 ELSE 0 END bili_SOFA_GT2,
					CASE WHEN c.crea_SOFA >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2,
					CASE WHEN c.cv_SOFA >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2,
					CASE WHEN GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) >= 2 THEN 1 ELSE 0 END resp_SOFA_GT2,
					CASE WHEN c.gcs_SOFA >= 2 THEN 1 ELSE 0 END gcs_SOFA_GT2,
					CASE WHEN c.urine_SOFA >= 2 THEN 1 ELSE 0 END urine_SOFA_GT2,
					CASE WHEN p.plat_SOFA >= 2 THEN 1 ELSE 0 END plat_SOFA_prior_GT2,
					CASE WHEN p.bili_SOFA >= 2 THEN 1 ELSE 0 END bili_SOFA_prior_GT2,
					CASE WHEN p.crea_SOFA >= 2 THEN 1 ELSE 0 END crea_SOFA_prior_GT2,
					CASE WHEN p.cv_SOFA >= 2 THEN 1 ELSE 0 END cv_SOFA_prior_GT2,
					CASE WHEN GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) >= 2 THEN 1 ELSE 0 END resp_SOFA_prior_GT2,
					CASE WHEN p.gcs_SOFA >= 2 THEN 1 ELSE 0 END gcs_SOFA_prior_GT2,
					CASE WHEN p.urine_SOFA >= 2 THEN 1 ELSE 0 END urine_SOFA_prior_GT2,
					CASE WHEN c.plat_SOFA - p.plat_SOFA >= 2 THEN 1 ELSE 0 END plat_SOFA_GT2_diff,
					CASE WHEN c.bili_SOFA - p.bili_SOFA >= 2 THEN 1 ELSE 0 END bili_SOFA_GT2_diff,
					CASE WHEN c.crea_SOFA - p.crea_SOFA >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2_diff,
					CASE WHEN c.cv_SOFA - p.cv_SOFA >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2_diff,
					CASE WHEN GREATEST(c.pao2_resp_SOFA, c.spo2_resp_SOFA) - GREATEST(p.pao2_resp_SOFA, p.spo2_resp_SOFA) >= 2 THEN 1 ELSE 0 END resp_SOFA_GT2_diff,
					CASE WHEN c.gcs_SOFA - p.gcs_SOFA >= 2 THEN 1 ELSE 0 END gcs_SOFA_GT2_diff,
					CASE WHEN c.urine_SOFA - p.urine_SOFA >= 2 THEN 1 ELSE 0 END urine_SOFA_GT2_diff,
					CASE WHEN c.vaso_shock - p.vaso_shock = 1 THEN 1 ELSE 0 END vaso_shock_diff,
					CASE WHEN c.map_shock - p.map_shock = 1 THEN 1 ELSE 0 END map_shock_diff,
					CASE WHEN c.lact_shock - p.lact_shock = 1 THEN 1 ELSE 0 END lact_shock_diff
					FROM 
						{suspected_infection} as cohort
					LEFT JOIN
						{sepsis_sofa} as c USING (person_id, admit_date)
					LEFT JOIN
						{sepsis_sofa}_prior as p USING (person_id, admit_date)
					ORDER BY
						person_id, CAST(admit_date AS DATE)
				'''
		if not format_query:
			pass
		else:
			query = query.format_map(
							{**{"sepsis_difference": self.config_dict["sepsis_difference"],
								"suspected_infection": self.config_dict["suspected_infection"],
							    "sepsis_sofa": self.config_dict["sepsis_sofa"]}})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query

	def get_score_query(self, format_query=True, prior=False):
		query = '''
				CREATE OR REPLACE TABLE `{sepsis_sofa}` AS
				SELECT 
                susp_inf_rollup.person_id, 
                CAST(susp_inf_rollup.admit_date AS DATE) AS admit_date, 
                CAST(discharge_date AS DATE) AS discharge_date,
                min_bc, 
                min_systemic_abx, 
                sepsis_index_date, 
                min_platelet, 
                CASE 
					WHEN min_platelet IS NULL THEN 0 
					WHEN min_platelet < 20 THEN 4 
					WHEN min_platelet < 50 THEN 3
					WHEN min_platelet < 100 THEN 2 
					WHEN min_platelet < 150 THEN 1 
					ELSE 0
                END plat_SOFA,
                max_bilirubin, 
                CASE 
                    WHEN max_bilirubin IS NULL THEN 0 
                    WHEN max_bilirubin >= 12 THEN 4 
                    WHEN max_bilirubin >= 6 THEN 3
                    WHEN max_bilirubin >= 2 THEN 2 
                    WHEN max_bilirubin >= 1.2 THEN 1 
                    ELSE 0
                END bili_SOFA,
                max_creatinine, 
                CASE 
					WHEN susp_inf_rollup.age_in_months >= 216 THEN #  look into creatinine issue
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 5 THEN 4 
							WHEN max_creatinine >= 3.5 THEN 3
							WHEN max_creatinine >= 2 THEN 2 
							WHEN max_creatinine >= 1.2 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >= 144 THEN
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 4.2 THEN 4 
							WHEN max_creatinine >= 2.9 THEN 3
							WHEN max_creatinine >= 1.7 THEN 2 
							WHEN max_creatinine >= 1.0 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >= 60 THEN
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 2.6 THEN 4 
							WHEN max_creatinine >= 1.8 THEN 3
							WHEN max_creatinine >= 1.1 THEN 2 
							WHEN max_creatinine >= 0.7 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >= 24 THEN
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 2.3 THEN 4 
							WHEN max_creatinine >= 1.6 THEN 3
							WHEN max_creatinine >= 0.9 THEN 2 
							WHEN max_creatinine >= 0.6 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >= 12 THEN
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 1.5 THEN 4 
							WHEN max_creatinine >= 1.1 THEN 3
							WHEN max_creatinine >= 0.6 THEN 2 
							WHEN max_creatinine >= 0.4 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >= 1 THEN
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 1.2 THEN 4 
							WHEN max_creatinine >= 0.8 THEN 3
							WHEN max_creatinine >= 0.5 THEN 2 
							WHEN max_creatinine >= 0.3 THEN 1 
							ELSE 0 
						END
					)
					ELSE
					(
						CASE
							WHEN max_creatinine IS NULL THEN 0 
							WHEN max_creatinine >= 1.6 THEN 4 
							WHEN max_creatinine >= 1.2 THEN 3
							WHEN max_creatinine >= 1.0 THEN 2 
							WHEN max_creatinine >= 0.8 THEN 1 
							ELSE 0 
						END
					)
				
                END crea_SOFA,
                max_dopamine_days,
				max_dobutamine_days,
				max_epinephrine_days,
				max_norepinephrine_days,
                min_map, 
                CASE 
					WHEN susp_inf_rollup.age_in_months >=216 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <70 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >=144 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <67 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >=60 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <65 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >=24 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <62 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >=12 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <60 THEN 1 
							ELSE 0 
						END
					)
					WHEN susp_inf_rollup.age_in_months >=1 THEN
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <55 THEN 1 
							ELSE 0 
						END
					)
					ELSE
					(
						CASE
							WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL THEN 3
							WHEN max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 2
							WHEN min_map <46 THEN 1 
							ELSE 0 
						END
					)
                END cv_SOFA,
                min_pao2fio2_ratio,
                min_spo2fio2_ratio,
                count_vent_mode, 
                CASE 
                    WHEN (count_vent_mode IS NOT NULL AND min_pao2fio2_ratio < 100) THEN 4
                    WHEN (count_vent_mode IS NOT NULL AND min_pao2fio2_ratio < 200) THEN 3 
                    WHEN min_paO2fiO2_ratio < 300 THEN 2 
                    WHEN min_paO2fiO2_ratio < 400 THEN 1
                    ELSE 0 
                END pao2_resp_SOFA,
				CASE 
                    WHEN (count_vent_mode IS NOT NULL AND min_spo2fio2_ratio < 148) THEN 4
                    WHEN (count_vent_mode IS NOT NULL AND min_spo2fio2_ratio < 221) THEN 3 
                    WHEN min_spo2fio2_ratio < 264 THEN 2 
                    WHEN min_spo2fio2_ratio < 292 THEN 1
                    ELSE 0 
                END spo2_resp_SOFA,
                min_gcs, 
                CASE 
                    WHEN min_gcs IS NULL THEN 0 
                    WHEN min_gcs < 6 THEN 4 
                    WHEN min_gcs < 10 THEN 3
                    WHEN min_gcs < 13 THEN 2 
                    WHEN min_gcs < 15 THEN 1 
                    ELSE 0 
                END gcs_SOFA,
                min_urine_daily, 
                CASE 
                    WHEN min_urine_daily IS NULL THEN 0 
                    WHEN min_urine_daily < 200 THEN 4 
                    WHEN min_urine_daily < 500 THEN 3 
                    ELSE 0 
                END urine_SOFA,
                CASE 
                    WHEN max_epinephrine_days IS NOT NULL or max_norepinephrine_days IS NOT NULL OR max_dopamine_days IS NOT NULL or max_dobutamine_days IS NOT NULL THEN 1 
                    ELSE 0 
                END vaso_shock,
                CASE 
                    WHEN min_map < 65 THEN 1 
                    ELSE 0 
                END map_shock,
                max_lactate, 
                CASE 
                    WHEN max_lactate > 2 THEN 1 
                    ELSE 0 
                END lact_shock
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN {sepsis_platelet} USING (person_id, admit_date)
            LEFT JOIN {sepsis_bilirubin} USING (person_id, admit_date)
            LEFT JOIN {sepsis_creatinine} USING (person_id, admit_date)
            LEFT JOIN {sepsis_dopamine} USING (person_id, admit_date)
			LEFT JOIN {sepsis_dobutamine} USING (person_id, admit_date)
			LEFT JOIN {sepsis_epinephrine} USING (person_id, admit_date)
			LEFT JOIN {sepsis_norepinephrine} USING (person_id, admit_date)
            LEFT JOIN {sepsis_map} USING (person_id, admit_date)
            LEFT JOIN {sepsis_pao2_fio2} USING (person_id, admit_date)
            LEFT JOIN {sepsis_vent} USING (person_id, admit_date)
            LEFT JOIN {sepsis_gcs} USING (person_id, admit_date)
            LEFT JOIN {sepsis_urine} USING (person_id, admit_date)
            LEFT JOIN {sepsis_lactate} USING (person_id, admit_date)
            LEFT JOIN {sepsis_spo2_fio2} USING (person_id, admit_date)
            ORDER BY person_id, CAST(admit_date AS DATE)
		'''
		
		if not format_query:
			pass
		else:
			query = query.format_map(
						{**{"suspected_infection":self.config_dict['suspected_infection'],
							"sepsis_platelet":self.config_dict['sepsis_platelet'] + '_prior' if prior else self.config_dict['sepsis_platelet'],
							"sepsis_creatinine":self.config_dict['sepsis_creatinine'] + '_prior' if prior else self.config_dict['sepsis_creatinine'],
							"sepsis_gcs":self.config_dict['sepsis_gcs'] + '_prior' if prior else self.config_dict['sepsis_gcs'],
							"sepsis_bilirubin":self.config_dict['sepsis_bilirubin'] + '_prior' if prior else self.config_dict['sepsis_bilirubin'],
							"sepsis_vent":self.config_dict['sepsis_vent'] + '_prior' if prior else self.config_dict['sepsis_vent'],
							"sepsis_lactate":self.config_dict['sepsis_lactate'] + '_prior' if prior else self.config_dict['sepsis_lactate'],
							"sepsis_spo2_fio2":self.config_dict['sepsis_spo2_fio2'] + '_prior' if prior else self.config_dict['sepsis_spo2_fio2'],
							"sepsis_dopamine":self.config_dict['sepsis_dopamine'] + '_prior' if prior else self.config_dict['sepsis_dopamine'],
							"sepsis_dobutamine":self.config_dict['sepsis_dobutamine'] + '_prior' if prior else self.config_dict['sepsis_dobutamine'],
							"sepsis_epinephrine":self.config_dict['sepsis_epinephrine'] + '_prior' if prior else self.config_dict['sepsis_epinephrine'],
							"sepsis_norepinephrine":self.config_dict['sepsis_norepinephrine'] + '_prior' if prior else self.config_dict['sepsis_norepinephrine'],
							"sepsis_map":self.config_dict['sepsis_map'] + '_prior' if prior else self.config_dict['sepsis_map'],
							"sepsis_urine":self.config_dict['sepsis_urine'] + '_prior' if prior else self.config_dict['sepsis_urine'],
							"sepsis_pao2_fio2":self.config_dict['sepsis_pao2_fio2'] + '_prior' if prior else self.config_dict['sepsis_pao2_fio2'],
						    "sepsis_sofa":self.config_dict["sepsis_sofa"] + '_prior' if prior else self.config_dict["sepsis_sofa"]}})
			
		if self.config_dict['print_query']:
			print(query)
			
		return query
	
	def run_score_queries(self):
		self.db.execute_sql(self.get_score_query())
		self.db.execute_sql(self.get_score_query(prior=True))
	
	def run_difference_query(self):
		self.db.execute_sql(self.get_difference_query())
		
	def create_sofa_tables(self):
		self.run_score_queries()
		self.run_difference_query()

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
			"save_sofa":True
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