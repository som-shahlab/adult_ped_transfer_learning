import os 
from ..database import Database

class Cohort:
    
    def __init__(self, *args, **kwargs):
        self.config = self.get_config(**kwargs)
        self.db = Database(**self.config)

    def get_base_query(self):
        """
        A reference to the data prior to transformation
        """
        raise NotImplementedError

    def get_transform_query(self):
        """
        A query that transforms the base query
        """
        raise NotImplementedError

    def get_create_query(self):
        """
        Constructs a create table query
        """
        raise NotImplementedError

    def create_cohort_table(self):
        """
        Creates the cohort table in the database
        """
        self.db.execute_sql(self.get_create_query())
            
    def get_defaults(self):
        return {
            "google_application_credentials": os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            ),
            "gcloud_project": "som-nero-nigam-starr",
            "dataset_project": None,
            "rs_dataset_project": None,
            "dataset": "starr_omop_cdm5_deid_20210723",
            "rs_dataset": "temp_dataset",
            "cohort_name": "temp_cohort",
            "limit": None,
            "min_stay_hour": None,
        }

    def override_defaults(self, **kwargs):
        return {**self.get_defaults(), **kwargs}

    def get_config(self, **kwargs):
        config = self.override_defaults(**kwargs)

        # Handle special parameters
        config["limit_str"] = (
            "LIMIT {}".format(config["limit"])
            if (
                (config["limit"] is not None)
                and (config["limit"] != "")
                and (config["limit"] != 0)
            )
            else ""
        )
        config["where_str"] = (
            "WHERE DATETIME_DIFF(visit_end_datetime, visit_start_datetime, hour)>{}".format(
                str(config["min_stay_hour"])
            )
            if (
                (config["min_stay_hour"] is not None)
                and (config["min_stay_hour"] != "")
            )
            else ""
        )
        config["dataset_project"] = (
            config["dataset_project"]
            if (
                (config["dataset_project"] is not None)
                and (config["dataset_project"] != "")
            )
            else config["gcloud_project"]
        )
        config["rs_dataset_project"] = (
            config["rs_dataset_project"]
            if (
                (config["rs_dataset_project"] is not None)
                and (config["rs_dataset_project"] != "")
            )
            else config["gcloud_project"]
        )
        return config
    
    
    def configure(self, **kwargs):
       self.config = {**self.config, **kwargs}