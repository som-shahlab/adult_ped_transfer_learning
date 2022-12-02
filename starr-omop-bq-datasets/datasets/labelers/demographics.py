from .base import LabelQuery


class AgeQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'Age group labels according to 1)pediatric age group and 2)intervals',
            "labeler_id":'age',
        }
    
    def get_base_query(self):
        return """
        WITH temp AS (
            SELECT t1.person_id, {window_start_field}, {window_end_field}
                ,DATE_DIFF({window_start_field}, birth_datetime, DAY) as age_days
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN {dataset_project}.{dataset}.person AS t2
                ON t1.person_id = t2.person_id
        )
        SELECT t1.*, age_days
            ,CASE 
                WHEN age_days BETWEEN 0 AND 27 THEN 'term neonatal'
                WHEN age_days BETWEEN 28 AND 365 THEN 'infancy'
                WHEN age_days BETWEEN 366 AND 2*365 THEN 'toddler'
                WHEN age_days BETWEEN 2*365+1 AND 5*365 THEN 'early childhood'
                WHEN age_days BETWEEN 5*365+1 AND 11*365 THEN 'middle childhood'
                WHEN age_days BETWEEN 11*365+1 AND 18*365 THEN 'early adolescence'
                WHEN age_days BETWEEN 18*365+1 AND 21*365 THEN 'late adolescence'
                WHEN age_days > 21*365 THEN 'non-pediatric'
                ELSE 'unknown'
            END as pediatric_age_group
            ,CASE 
                WHEN age_days BETWEEN 0 AND 17*365 THEN '[0,18)'
                WHEN age_days BETWEEN 17*365+1 AND 29*365 THEN '[18,30)'
                WHEN age_days BETWEEN 29*365+1 AND 39*365 THEN '[30,40)'
                WHEN age_days BETWEEN 39*365+1 AND 49*365 THEN '[40,50)'
                WHEN age_days BETWEEN 49*365+1 AND 59*365 THEN '[50,60)'
                WHEN age_days BETWEEN 59*365+1 AND 69*365 THEN '[60,70)'
                WHEN age_days BETWEEN 69*365+1 AND 79*365 THEN '[70,80)'
                WHEN age_days BETWEEN 79*365+1 AND 89*365 THEN '[80,90)'
                WHEN age_days > 89*365 THEN '[90,)'
                ELSE 'unknown'
            END as age_group
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN temp USING (person_id, {window_start_field}, {window_end_field})
        """


class SexQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'OMOP standard concepts for sex',
            "labeler_id":'sex',
        }
    
    def get_base_query(self):
        return """
        SELECT t1.person_id, {window_start_field}, {window_end_field}
            ,t3.concept_name as sex
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN {dataset_project}.{dataset}.person AS t2
            ON t1.person_id = t2.person_id
        LEFT JOIN {dataset_project}.{dataset}.concept AS t3 
            ON t2.gender_concept_id=t3.concept_id
        """
        


    
class RaceQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'OMOP standard concepts for race',
            "labeler_id":'race',
        }
    
    def get_base_query(self):
        return """
        SELECT t1.person_id, {window_start_field}, {window_end_field}
            ,t3.concept_name as race
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN {dataset_project}.{dataset}.person AS t2
            ON t1.person_id = t2.person_id
        LEFT JOIN {dataset_project}.{dataset}.concept AS t3 
            ON t2.race_concept_id=t3.concept_id
        """