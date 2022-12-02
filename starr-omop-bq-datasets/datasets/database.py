import pandas as pd
import os
import shutil

import google.auth
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

from datasets.util import overwrite_dir, yaml_read


class Database:
    """
    A class defining a BigQuery Database
    """

    def __init__(self, **kwargs):

        self.config = self.override_defaults(**kwargs)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config[
            "google_application_credentials"
        ]
        os.environ["GCLOUD_PROJECT"] = self.config["gcloud_project"]

        # https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas
        credentials, your_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = bigquery.Client(credentials=credentials, project=your_project_id)
        self.bqstorageclient = BigQueryReadClient(
            credentials=credentials
        )

    def get_defaults(self):
        """
        Defaults for values in the config
        """
        return {
            "gcloud_project": "som-nero-nigam-starr",
            "google_application_credentials": os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            ),
        }

    def override_defaults(self, **kwargs):
        return {**self.get_defaults(), **kwargs}

    def read_sql_query(
        self,
        query,
        dialect="standard",
        use_bqstorage_api=True,
        progress_bar_type=None,
        **kwargs
    ):
        """
        Read a sql query directly into a pandas DataFrame.
        Uses default project defined at instantiation.
        Args:
            query: A SQL query as a string
            dialect: BigQuery dialect to use. Default "standard"
            use_bq_storage_api: Whether to use the BigQuery Storage API
        """
        df = pd.read_gbq(
            query,
            project_id=self.config["gcloud_project"],
            dialect=dialect,
            use_bqstorage_api=use_bqstorage_api,
            progress_bar_type=progress_bar_type,
            **kwargs
        )
        return df

    def stream_query(self, query, output_path, overwrite=False, combine_every=1000):
        """
        Streams a query to pandas dataframes in chunks using the Storage API.
        Results will be written as parquet files of size 1024*`combine_every` rows
        query: SQL query to execute
        output_path: a directory to write the result
        overwrite: Whether to overwrite output_path
        combine_every: The number of chunks to combine before writing a file
        """
        result = (
            self.client.query(query)
            .result(
                page_size=1024
            )  # page_size doesn't seem to do anything if using bqstorage_client?
            .to_dataframe_iterable(bqstorage_client=self.bqstorageclient)
        )
        result_dict = {}
        for i, rows in enumerate(result):
            if i == 0:
                overwrite_dir(output_path, overwrite=overwrite)
            result_dict[i] = rows
            if (i % combine_every == 0) & (i > 0):
                result_df = pd.concat(result_dict, ignore_index=True)
                result_df.to_parquet(
                    os.path.join(output_path, "features_{i}.parquet".format(i=i)),
                    engine="pyarrow",
                )
                result_dict = {}
        if len(list(result_dict.keys())) > 0:
            result_df = pd.concat(result_dict, ignore_index=True)
            result_df.to_parquet(
                os.path.join(output_path, "features_{i}.parquet".format(i=i)),
                engine="pyarrow",
            )

    def execute_sql(self, query):
        """
        Executes sql statement
        """
        return self.client.query(query).result()

    def execute_sql_to_destination_table(self, query, destination=None, **kwargs):
        """
        Executes a query and writes the result to a destination table
        """
        if destination is None:
            raise ValueError("destination must not be None")

        self.client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                destination=destination, write_disposition="WRITE_TRUNCATE"
            ),
        ).result()