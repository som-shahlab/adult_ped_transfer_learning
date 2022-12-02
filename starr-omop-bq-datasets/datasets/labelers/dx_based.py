from .base import LabelQuery

class DxLabelQuery(LabelQuery):
    
    def get_base_query(self):
        return """
        WITH concepts AS 
        (
            SELECT DISTINCT concept_id 
            FROM (
                SELECT concept_id 
                FROM `{dataset_project}.{dataset}.concept` 
                WHERE concept_id IN ({condition_concept_ids})

                UNION DISTINCT 

                SELECT c.concept_id
                FROM `{dataset_project}.{dataset}.concept` c
                INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                  ON c.concept_id = ca.descendant_concept_id
                  AND ca.ancestor_concept_id in ({condition_concept_ids})
                  AND c.invalid_reason is null
            )
        ),
        all_condition_occurrences AS 
        (
            SELECT co.person_id
                ,co.condition_concept_id
                ,co.condition_start_DATETIME
            FROM `{dataset_project}.{dataset}.condition_occurrence` co
            INNER JOIN concepts c ON co.condition_concept_id=c.concept_id 
        ),
        condition_occurrences_in_window AS 
        (
            SELECT t1.*
                ,co.condition_concept_id 
                ,co.condition_start_DATETIME
                ,ROW_NUMBER() OVER(
                    PARTITION BY t1.person_id, t1.{window_start_field}, t1.{window_end_field} 
                    ORDER BY co.condition_start_datetime
                ) AS rn
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1 
            INNER JOIN all_condition_occurrences co 
                ON t1.person_id=co.person_id 
                AND co.condition_start_datetime >= t1.{window_start_field} 
                AND co.condition_start_datetime <= t1.{window_end_field}
        )
        SELECT t1.* 
            ,CASE 
                WHEN condition_concept_id IS NOT NULL THEN 1 
                ELSE 0 
            END AS {labeler_id}_label
            ,condition_start_DATETIME AS {labeler_id}_start_datetime
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1 
        LEFT JOIN condition_occurrences_in_window co 
            ON t1.person_id = co.person_id
            AND t1.{window_start_field} = co.{window_start_field}
            AND t1.{window_end_field} = co.{window_end_field}
            AND co.rn = 1
        """
    
class HypoglycemiaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'hypoglycemia defined as at least one condition occurrence of "hypoglycemia"',
             "labeler_id":'hypoglycemia_dx',
             "condition_concept_ids": [
                 380688,4226798,36714116,24609,
                 4029423,45757363,4096804,4048805,
                 4228112,23034,4029424,45769876
             ]
         }
        

class AKIDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'AKI defined as having at least one condition occurrence of "acute renal failure"',
             "labeler_id":'aki_dx',
             "condition_concept_ids": [
                 197320,432961,444044
             ]
         }
        

class AnemiaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'anemia defined as having at least one condition occurrence of "anemia"',
             "labeler_id":'anemia_dx',
             "condition_concept_ids": [
                 439777, 37018722, 37017132, 35624756, 
                 4006467, 37398911, 37395652
             ]
         }
        
        
class HyperkalemiaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'hyperkalemia defined as having at least one condition occurrence of "hyperkalemia"',
             "labeler_id":'hyperkalemia_dx',
             "condition_concept_ids": [
                 434610
             ]
         }
        
        
class HyponatremiaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'hyponatremia defined as having at least one condition occurrence of "Hyponatremia"',
             "labeler_id":'hyponatremia_dx',
             "condition_concept_ids": [
                 435515,4232311
             ]
         }
        
        
class ThrombocytopeniaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'thrombocytopenia defined as having at least one condition occurrence of "Thrombocytopenia"',
             "labeler_id":'thrombocytopenia_dx',
             "condition_concept_ids": [
                 432870
             ]
         }
        
        
class NeutropeniaDxQuery(DxLabelQuery):
    def get_query_config(self):
         return {
             "labeler_info": 'neutropenia defined as having at least one condition occurrence of "Neutropenia"',
             "labeler_id":'neutropenia_dx',
             "condition_concept_ids": [
                 301794,320073
             ]
         }