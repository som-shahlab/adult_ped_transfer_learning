from .base import LabelQuery


class HyperkalemiaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for hyperkalemia using blood potassium concentration (mmol/L). Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range.',
            "labeler_id":'hyperkalemia_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (40653595, 37074594, 40653596)

            UNION DISTINCT

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (40653595, 37074594, 40653596)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
            SELECT
              t1.*, m.measurement_datetime
             ,CASE 
                  WHEN m.unit_concept_id = 8840 then m.value_as_number / 18
                  ELSE m.value_as_number 
              END AS value_as_number
              ,CASE 
                  WHEN m.unit_concept_id = 8840 then m.range_low / 18
                  ELSE m.range_low 
              END AS range_low
              ,CASE 
                  WHEN m.unit_concept_id = 8840 then m.range_high / 18
                  ELSE m.range_high 
              END AS range_high
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              on t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                    8753, -- mmol/L
                    9557, --mEq/L (1-to-1 -> mmol/L)
                    8840 --mg/dl (divide by 18 -> mmol/L)
              )
            INNER JOIN measurement_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
        ),
        max_measurements as 
        (
            SELECT person_id
                ,{window_start_field}, {window_end_field}
                ,max(value_as_number) as max_potassium
            FROM all_measurements
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,range_low,range_high
                ,case when value_as_number > 5.5 then 1 else 0 end as mild 
                ,case when value_as_number > 6.0 then 1 else 0 end as moderate 
                ,case when value_as_number > 7 then 1 else 0 end as severe
                ,case when value_as_number > range_high then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number > 5.5 OR value_as_number > range_high)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,max_potassium as {labeler_id}_max_potassium
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_high as {labeler_id}_abnormal_threshold
        FROM max_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """
        
        
class HypoglycemiaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for hypoglycemia using blood glucose concentration (mmol/L). Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range.',
            "labeler_id":'hypoglycemia_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (4144235, 1002597)

            UNION DISTINCT

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (4144235, 1002597)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
            SELECT
              t1.*, m.measurement_datetime
              ,CASE
                  WHEN m.unit_concept_id IN (8840, 9028) THEN m.value_as_number / 18 
                  ELSE m.value_as_number
              END AS value_as_number
              ,CASE
                  WHEN m.unit_concept_id IN (8840, 9028) THEN m.range_low / 18 
                  ELSE m.range_low
              END AS range_low
              ,CASE
                  WHEN m.unit_concept_id IN (8840, 9028) THEN m.range_high / 18 
                  ELSE m.range_high
              END AS range_high
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              on t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                  8840, -- mg/dL
                  9028, -- mg/dL calculated
                  8753 -- mmol/L (x 18 to get mg/dl)
              )
            INNER JOIN measurement_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
        ),
        min_measurements as 
        (
            SELECT person_id
                ,{window_start_field}, {window_end_field}
                ,min(value_as_number) as hg_min_glucose
            FROM all_measurements
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,range_low,range_high
                ,case when value_as_number <= 3.9 then 1 else 0 end as mild 
                ,case when value_as_number < 3.5 then 1 else 0 end as moderate 
                ,case when value_as_number < 3 then 1 else 0 end as severe
                ,case when value_as_number < range_low then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number <= 3.9 OR value_as_number < range_low)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,hg_min_glucose as {labeler_id}_min_glucose
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_low as {labeler_id}_abnormal_threshold
        FROM min_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """


class AcuteKidneyInjuryQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for acute kidney injury based on blood creatinine levels (umol/L) according to KDIGO (stages 1,2, and 3), and abnormal range.',
            "labeler_id":'aki_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (37029387,4013964,2212294,3051825)

            UNION DISTINCT 

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (37029387,4013964,2212294,3051825)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
        SELECT
          t1.*, m.measurement_datetime
          ,case 
              when unit_concept_id = 8840 then value_as_number / 0.0113122 
              when unit_concept_id = 8837 then value_as_number * 0.001 / 0.0113122 
          else value_as_number end as value_as_number
          ,case 
              when unit_concept_id = 8840 then range_high / 0.0113122 
              when unit_concept_id = 8837 then range_high * 0.001 / 0.0113122 
          else range_high end as range_high
          ,case 
              when unit_concept_id = 8840 then range_low / 0.0113122 
              when unit_concept_id = 8837 then range_low * 0.001 / 0.0113122 
          else range_low end as range_low
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
        LEFT JOIN {dataset_project}.{dataset}.measurement m
            ON t1.person_id = m.person_id
            AND m.unit_concept_id IN (
                8749,  -- umol/l (x0.0113122 to get mg/dl)
                8840,  -- mg/dl
                8837   -- ug/dl (x0.001 to get mg/dl)
              )
        INNER JOIN measurement_concepts mc 
            ON m.measurement_concept_id = mc.concept_id
        ),
        -- GET BASELINE CREATININE MEASUREMENTS (MIN MEASUREMENT BETWEEN ADMISSION TIME AND 3 MONTHS PRIOR)
        base_3month as (
            SELECT person_id,{window_start_field}, {window_end_field}
                ,MIN(m.value_as_number) as value_as_number
            FROM all_measurements m 
            WHERE measurement_datetime >= date_add({window_start_field}, INTERVAL -90 day)
                AND measurement_datetime < {window_start_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        base_by_age_norm as (
            SELECT m.person_id,{window_start_field},{window_end_field}
                ,CASE 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN 0 AND 14 THEN 0.92 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN 15 AND (2*365-1) THEN 0.36 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN (2*365) AND (5*365-1) THEN 0.43 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN (5*365) AND (12*365-1) THEN 0.61 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN (12*365) AND (15*365-1) THEN 0.81 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) BETWEEN (15*365) AND (19*365-1) THEN 0.84 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) >= (19*365) AND gender_concept_id=8532 THEN 1.1 / 0.0113122 
                    WHEN DATE_DIFF({window_start_field}, birth_datetime, DAY) >= (19*365) AND gender_concept_id<>8532 THEN 1.2 / 0.0113122
                    ELSE NULL
                END AS value_as_number
            FROM (SELECT DISTINCT person_id, {window_start_field}, {window_end_field} FROM all_measurements) m
            LEFT JOIN `{dataset_project}.{dataset}.person` person
                ON m.person_id=person.person_id
        ),
        base_measurements as (
            SELECT b_age.person_id, b_age.{window_start_field}, b_age.{window_end_field} 
                ,COALESCE(b_3month.value_as_number, b_age.value_as_number) as value_as_number
            FROM base_by_age_norm b_age 
            LEFT JOIN base_3month b_3month 
                ON b_age.person_id=b_3month.person_id 
                AND b_age.{window_start_field}=b_3month.{window_start_field}
                AND b_age.{window_end_field}=b_3month.{window_end_field}
        ),
        max_measurements as (
            SELECT m.person_id, {window_start_field}, {window_end_field}
                ,MAX(m.value_as_number) as aki_max_creatinine
            FROM all_measurements m 
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,m.value_as_number
                ,range_low,range_high
                ,case when (
                    m.value_as_number/b.value_as_number >= 1.5
                    OR m.value_as_number-b.value_as_number >= 26.52
                ) then 1 else 0 end as mild 
                ,case when (
                    m.value_as_number/b.value_as_number >= 2.0
                ) then 1 else 0 end as moderate 
                ,case when (
                    m.value_as_number/b.value_as_number >= 3.0 
                    OR m.value_as_number-b.value_as_number >= 353.6
                ) then 1 else 0 end as severe
                ,case when m.value_as_number > range_high then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements m
            LEFT JOIN base_measurements b USING (person_id, {window_start_field}, {window_end_field})
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field}  
                AND (
                    (
                        m.value_as_number/b.value_as_number >= 1.5
                        OR m.value_as_number-b.value_as_number >= 26.52
                    )
                    OR
                    (
                        m.value_as_number > range_high
                    )
                )
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,aki_max_creatinine as {labeler_id}_max_creatinine
            ,first_mild.value_as_number as {labeler_id}_aki1_measurement
            ,first_mild.measurement_datetime as {labeler_id}_aki1_measurement_datetime
            ,mild as {labeler_id}_aki1_label
            ,first_moderate.value_as_number as {labeler_id}_aki2_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_aki2_measurement_datetime
            ,moderate as {labeler_id}_aki2_label
            ,first_severe.value_as_number as {labeler_id}_aki3_measurement
            ,first_severe.measurement_datetime as {labeler_id}_aki3_measurement_datetime
            ,severe as {labeler_id}_aki3_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_high as {labeler_id}_abnormal_threshold
        FROM max_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """
    
    
class HyponatremiaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for hyponatremia based on blood sodium concentration (mmol/L). Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range.',
            "labeler_id":'hyponatremia_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (40653762)

            UNION DISTINCT

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (40653762)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
            SELECT
              t1.*, m.measurement_datetime
             ,m.value_as_number
             ,range_low, range_high
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              on t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                8753, -- mmol/L
                9557 --mEq/L (1-to-1 -> mmol/L)
              )
            INNER JOIN measurement_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
        ),
        min_measurements as 
        (
            SELECT person_id
                ,{window_start_field}, {window_end_field}
                ,min(value_as_number) as hn_min_sodium
            FROM all_measurements
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,range_low,range_high
                ,case when value_as_number <= 135 then 1 else 0 end as mild 
                ,case when value_as_number < 130 then 1 else 0 end as moderate 
                ,case when value_as_number < 125 then 1 else 0 end as severe
                ,case when value_as_number < range_low then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number <= 135 OR value_as_number < range_low)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,hn_min_sodium as {labeler_id}_min_sodium
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_low as {labeler_id}_abnormal_threshold
        FROM min_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """
    
    
class AnemiaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for anemia based on hemoglobin levels (g/L). Thresholds: mild(<120),moderate(<110),severe(<70), and reference range',
            "labeler_id":'anemia_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (37072252)

            UNION DISTINCT

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (37072252)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
            SELECT
              t1.*, m.measurement_datetime
             ,case 
                  when m.unit_concept_id = 8840 then m.value_as_number / 100
                  else m.value_as_number * 10
              end as value_as_number
             ,case 
                  when m.unit_concept_id = 8840 then m.range_high / 100
                  else m.range_high * 10
              end as range_high
              ,case 
                  when m.unit_concept_id = 8840 then m.range_low / 100
                  else m.range_low * 10
              end as range_low
               
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              on t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                8713 -- g / dL
                ,8840 -- mg / dL (divide by 1000 to get g/dL)
              )
            INNER JOIN measurement_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
        ),
        min_measurements as 
        (
            SELECT person_id
                ,{window_start_field}, {window_end_field}
                ,min(value_as_number) as anemia_min_hgb
            FROM all_measurements
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field} 
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,range_low,range_high
                ,case when value_as_number < 120 then 1 else 0 end as mild 
                ,case when value_as_number < 110 then 1 else 0 end as moderate 
                ,case when value_as_number < 70 then 1 else 0 end as severe
                ,case when value_as_number < range_low then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number < 120 OR value_as_number < range_low)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,anemia_min_hgb as {labeler_id}_min_hgb
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_low as {labeler_id}_abnormal_threshold
        FROM min_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """
    
    
class NeutropeniaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for neutropenia based on neutrophils count (thousands/uL). Thresholds: mild(<1.5), moderate(<1), severe(<0.5).',
            "labeler_id":'neutropenia_lab',
        }
    
    def get_base_query(self):
        return """
        -- WBC (LEUKOCYTES)
        WITH wbc_concepts as 
        (
            -- Use 3010813
            -- 3000905 has unusual distribution
            SELECT 
                c.concept_id, 'wbc' as concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            WHERE concept_id = 3010813       
        ),
        all_wbc as 
        (
          SELECT t1.*
              ,m.measurement_datetime
              ,m.value_as_number as wbc
          FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
          LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              ON t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                8848,  -- 1000/uL
                8961  -- 1000/mm^3, equivalent to 8848
              )
          INNER JOIN wbc_concepts mc 
              on m.measurement_concept_id = mc.concept_id
        ),
        -- BANDS
        bands_concepts as 
        (
            -- 3035839 band form /100 leukocytes (%)
            -- 3018199 band form neutrophils in blood (count) 
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            WHERE concept_id in (3035839, 3018199)
        ),
        all_bands as
        (
            SELECT t1.*
                ,m.measurement_datetime 
                ,CASE 
                    WHEN m.measurement_concept_id = 3035839 AND m.value_as_number<=100
                      THEN m.value_as_number / 100 * wbc.wbc
                    WHEN m.measurement_concept_id = 3018199 AND m.unit_concept_id = 8784 
                      THEN m.value_as_number / 1000
                    ELSE NULL
                END AS bands_count
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
                ON t1.person_id = m.person_id
            INNER JOIN bands_concepts mc 
                ON m.measurement_concept_id = mc.concept_id
            LEFT JOIN all_wbc wbc 
                ON m.person_id = wbc.person_id
                AND m.measurement_datetime = wbc.measurement_datetime
        ),
        -- NEUTROPHILS
        neutrophils_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            WHERE concept_id in (37045722, 37049637)

            UNION DISTINCT 

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (37045722, 37049637)
                AND c.invalid_reason is null 
        ),
        all_neutrophils as 
        (
            SELECT t1.*
              ,m.measurement_datetime 
              ,CASE 
                  -- neutrophils /100 leukocytes
                  WHEN m.unit_concept_id = 8554 AND m.value_as_number<=100
                      THEN m.value_as_number / 100 * wbc.wbc
                  WHEN m.unit_concept_id = 8554 AND m.value_as_number>100
                      THEN NULL
                  WHEN m.unit_concept_id = 8784  
                      THEN m.value_as_number / 1000
                  ELSE m.value_as_number
              END AS neutrophils_count
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              ON t1.person_id = m.person_id
              AND m.unit_concept_id in (
                  8848, -- 1000/uL
                  8554, -- %
                  8784, -- cells/uL
                  8961 -- 1000/mm^3
              )
            INNER JOIN neutrophils_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
            LEFT JOIN all_wbc wbc
              ON m.person_id = wbc.person_id 
              AND m.measurement_datetime = wbc.measurement_datetime
        ),
        all_measurements AS
        (
            SELECT DISTINCT *
            FROM all_neutrophils
            FULL OUTER JOIN all_bands USING (person_id, {window_start_field}, {window_end_field}, measurement_datetime)
            FULL OUTER JOIN all_wbc USING (person_id, {window_start_field}, {window_end_field}, measurement_datetime)
            WHERE 
                (
                    neutrophils_count IS NOT NULL
                    OR bands_count IS NOT NULL
                    OR wbc IS NOT NULL
                ) 
        ),
        all_measurements_transformed AS
        (
            SELECT person_id, {window_start_field}, {window_end_field}, measurement_datetime
                ,CASE
                    WHEN neutrophils_count IS NOT NULL or bands_count IS NOT NULL 
                        THEN IFNULL(neutrophils_count,0) + IFNULL(bands_count,0)
                    WHEN wbc IS NOT NULL AND neutrophils_count IS NULL AND bands_count IS NULL
                        THEN wbc
                END AS value_as_number
            FROM all_measurements
        ),
        min_measurements AS
        (
            SELECT person_id, {window_start_field}, {window_end_field}
                ,MIN(value_as_number) AS np_min_neutrophils
            FROM all_measurements_transformed
            WHERE measurement_datetime >= {window_start_field} 
                AND measurement_datetime <= {window_end_field}
            GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,case when value_as_number < 1.5 then 1 else 0 end as mild 
                ,case when value_as_number < 1 then 1 else 0 end as moderate 
                ,case when value_as_number < 0.500 then 1 else 0 end as severe
                ,measurement_datetime
            FROM all_measurements_transformed
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number < 1.5)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,np_min_neutrophils as {labeler_id}_min_neutrophils
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
        FROM min_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        """

    
class ThrombocytopeniaQuery(LabelQuery):
    def get_query_config(self):
        return {
            "labeler_info":'lab-based definition for thrombocytopenia based on platelet count (10^9/L). Thresholds: mild (<150), moderate(<100), severe(<50), and reference range.',
            "labeler_id":'thrombocytopenia_lab',
        }
    
    def get_base_query(self):
        return """
        WITH measurement_concepts as 
        (
            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c 
            WHERE c.concept_id in (37037425,40654106)

            UNION DISTINCT

            SELECT 
                c.concept_id, concept_name
            FROM `{dataset_project}.{dataset}.concept` c
            INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
                ON c.concept_id = ca.descendant_concept_id
                AND ca.ancestor_concept_id in (37037425,40654106)
                AND c.invalid_reason is null 
        ),
        all_measurements AS
        (
            SELECT
              t1.*, m.measurement_datetime
             ,m.value_as_number
             ,m.range_low 
             ,m.range_high
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
            LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
              on t1.person_id = m.person_id
              AND m.unit_concept_id IN (
                8848 -- thousands / uL
                ,8961 -- thousands / mm^3 (equivalent to 8848)
              )
            INNER JOIN measurement_concepts mc 
              ON m.measurement_concept_id = mc.concept_id
        ),
        min_measurements as 
        (
            SELECT person_id
            ,{window_start_field}, {window_end_field}
            ,min(value_as_number) as tc_min_platelet
        FROM all_measurements
        WHERE measurement_datetime >= {window_start_field} 
            AND measurement_datetime <= {window_end_field} 
        GROUP BY person_id, {window_start_field}, {window_end_field}
        ),
        abnormal_measurements as 
        (
            SELECT person_id 
                ,{window_start_field}, {window_end_field}
                ,value_as_number
                ,range_low,range_high
                ,case when value_as_number < 150 then 1 else 0 end as mild 
                ,case when value_as_number < 100 then 1 else 0 end as moderate 
                ,case when value_as_number < 50 then 1 else 0 end as severe
                ,case when value_as_number < range_low then 1 else 0 end as abnormal_range
                ,measurement_datetime
            FROM all_measurements
            WHERE measurement_datetime >= admit_date 
                AND measurement_datetime <= discharge_date 
                AND (value_as_number < 150 OR value_as_number < range_low)
        ),
        mild as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, mild
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE mild = 1
        ), 
        moderate as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, moderate
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE moderate = 1
        ),
        severe as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, severe
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
            FROM abnormal_measurements 
            WHERE severe = 1
        ),
        abnormal_range as (
            SELECT 
                person_id, {window_start_field}, {window_end_field}
                ,value_as_number, measurement_datetime, abnormal_range
                ,ROW_NUMBER() OVER(
                    PARTITION BY person_id,{window_start_field}, {window_end_field} 
                    ORDER BY measurement_datetime
                    ) AS rn
                ,range_low, range_high
            FROM abnormal_measurements 
            WHERE abnormal_range = 1
        ),
        first_mild as (select * from mild where rn = 1),
        first_moderate as (select * from moderate where rn = 1),
        first_severe as (select * from severe where rn = 1),
        first_abnormal_range as (select * from abnormal_range where rn = 1)
        SELECT person_id, {window_start_field}, {window_end_field} 
            ,tc_min_platelet as {labeler_id}_min_platelet
            ,first_mild.value_as_number as {labeler_id}_mild_measurement
            ,first_mild.measurement_datetime as {labeler_id}_mild_measurement_datetime
            ,mild as {labeler_id}_mild_label
            ,first_moderate.value_as_number as {labeler_id}_moderate_measurement
            ,first_moderate.measurement_datetime as {labeler_id}_moderate_measurement_datetime
            ,moderate as {labeler_id}_moderate_label
            ,first_severe.value_as_number as {labeler_id}_severe_measurement
            ,first_severe.measurement_datetime as {labeler_id}_severe_measurement_datetime
            ,severe as {labeler_id}_severe_label
            ,first_abnormal_range.value_as_number as {labeler_id}_abnormal_measurement
            ,first_abnormal_range.measurement_datetime as {labeler_id}_abnormal_measurement_datetime
            ,abnormal_range as {labeler_id}_abnormal_label
            ,range_low as {labeler_id}_abnormal_threshold
        FROM min_measurements 
        FULL OUTER JOIN first_mild using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_moderate using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_severe using (person_id, {window_start_field}, {window_end_field})
        FULL OUTER JOIN first_abnormal_range using (person_id, {window_start_field}, {window_end_field})
        """