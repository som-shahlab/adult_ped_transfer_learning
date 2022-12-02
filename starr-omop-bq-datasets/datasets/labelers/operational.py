from .base import LabelQuery


class MortalityQuery(LabelQuery):
    """
    Notes:
        - There are cases in which the death date occurred before the admission date
        - Important to remove such patients from the dataset
    """
    def get_query_config(self):
        return {
            "labeler_info":'1 if death occured within the specified time window, 0 otherwise.',
            "labeler_id":'mortality',
        }
    
    def get_base_query(self):
        return """
        WITH temp AS (
            SELECT t1.person_id, {window_start_field}, {window_end_field}, death_date,
            CASE
                WHEN death_date BETWEEN CAST({window_start_field} AS DATE) AND CAST({window_end_field} AS DATE) THEN 1
                ELSE 0
            END as mortality_label
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            RIGHT JOIN {dataset_project}.{dataset}.death AS t2
                ON t1.person_id = t2.person_id
        )
        SELECT t1.*, IFNULL(mortality_label, 0) as mortality_label, death_date
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN temp USING (person_id, {window_start_field}, {window_end_field})
        """
        

class LOS7Query(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'1 if length of the specified time window is at least 7 days, 0 otherwise',
            "labeler_id":'los_7',
        }
    
    def get_base_query(self):
        return """
        WITH temp AS (
            SELECT *,
            DATE_DIFF(t1.{window_end_field}, t1.{window_start_field}, DAY) AS los_days
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        )
        SELECT *, 
        CAST(los_days >= 7 AS INT64) as los_7_label
        FROM temp
        """
    

class Readmission30Query(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'1 if readmission occurred within 30 days from the end of the specified time window, 0 otherwise',
            "labeler_id":'readmission_30',
        }
        
    def get_base_query(self):
         return """
        WITH temp AS (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY {window_start_field}) as row_number
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name}
        ),
        temp_shifted AS ( -- row shift by one
            SELECT person_id, {window_start_field}, {window_end_field}, row_number - 1 as row_number
            FROM temp
        ),
        temp_readmission_window AS ( --compare discharges from temp to admits from temp_shifted
            SELECT t1.person_id, 
                t1.{window_end_field}, 
                t2.{window_start_field}, 
                DATE_DIFF(t2.{window_start_field}, t1.{window_end_field}, DAY) AS readmission_window,
                t1.row_number
            FROM temp as t1
            INNER JOIN temp_shifted as t2
            ON t1.person_id = t2.person_id AND t1.row_number = t2.row_number
        ),
        result AS (
            SELECT t1.person_id, t1.{window_start_field}, t1.{window_end_field}, t2.readmission_window, 
            CASE 
                WHEN readmission_window BETWEEN 0 AND 30 THEN 1
                ELSE 0
            END as readmission_30_label
            FROM temp as t1
            INNER JOIN temp_readmission_window as t2
            on t1.person_id = t2.person_id AND t1.row_number = t2.row_number
        )
        SELECT t1.*, IFNULL(readmission_30_label, 0) as readmission_30_label, readmission_window
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN result USING (person_id, {window_start_field}, {window_end_field})
        """
        
        
class ICUAdmissionQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'1 if admitted to ICU during specified time window, 0 otherwise',
            "labeler_id":'icu_admission',
        }
    
    def get_base_query(self):
        return """
            WITH icu1 AS
            (
             SELECT
              t1.*,
              detail.visit_detail_start_datetime AS icu_start_datetime,
            FROM
              {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN
              {dataset_project}.{dataset}.visit_detail detail
            ON
              detail.person_id = t1.person_id
              AND visit_detail_start_datetime BETWEEN t1.{window_start_field} AND t1.{window_end_field}
              AND visit_detail_source_value IN 
               ('J4|J4|J4|',
                'J2|J2|J2|',
                'K4|K4|K4|',
                'M4|M4|M4|',
                'L4|L4|L4|',
                'ACA6 ICU|ACA6ICU|ACA6ICU|',
                'E2-ICU|E2|E2-ICU|Intensive Care',
                'VCP CCU 2|VCPC2|VCP CCU 2|Critical Care Medicine',
                'VCP CCU 1|VCPC1|VCP CCU 1|Critical Care Medicine',
                'D2ICU-SURGE|D2ICU|D2ICU|Intensive Care',
                '2 NORTH|2NCVICU|Cardiovascular Intensive Care|',
                'CVICU 220|CVICU220|Cardiovascular Intensive Care 220|', 
                'CVICU 320|CVICU320|Cardiovascular Intensive Care 320|',
                'CVICU|CVICU|Cardiovascular Intensive Care|',
                'E29-ICU|NICU|E29-ICU|Intensive Care',
                'NICU 260|NICU260|Neonatal Intensive Care|Neonatology', 
                'NICU 270|NICU270|Neonatal Intensive Care|Neonatology',
                'PICU 320|PICU320|Pediatric Intensive Care 320|',
                'PICU 420|PICU420|Pediatric Intensive Care 420|',
                'PICU|PICU|Pediatric Intensive Care|',
                'VCP NICU|VCPNICU|VCP NICU|Neonatology'
                )
            ),
            icu2 AS
            (
              SELECT
               icu1.*,
               RANK() OVER(
                   PARTITION BY icu1.person_id, icu1.{window_start_field}, icu1.{window_end_field}
                   ORDER BY icu1.icu_start_datetime ASC
               ) rank_
              FROM
               icu1
            ),
            icu3 AS 
            (
              SELECT 
               icu2.*,
              FROM 
               icu2 
              WHERE rank_ = 1
            )
            SELECT t1.*, CASE WHEN icu_start_datetime IS NULL THEN 0 ELSE 1 END as icu_admission_label, icu_start_datetime
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN icu3 USING (person_id, {window_start_field}, {window_end_field})
        """
    