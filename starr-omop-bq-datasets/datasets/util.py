import numpy as np
import pandas as pd
import yaml
import os
import shutil
import argparse
import pickle

def str2bool(v):
    """
    Converts strings to booleans (e.g., 't' -> True)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def yaml_write(x, path):
    """
    Writes yaml to disk
    """
    with open(path, "w") as fp:
        yaml.dump(x, fp)


def yaml_read(path):
    """
    Reads yaml from disk
    """
    with open(path, "r") as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def df_dict_concat(df_dict, outer_index_name="outer_index", drop_outer_index=False):
    """
    Concatenate a dictionary of dataframes together and remove the inner index
    """
    if isinstance(outer_index_name, str):
        reset_level = 1
    else:
        reset_level = len(outer_index_name)

    return (
        pd.concat(df_dict, sort=False)
        .reset_index(level=reset_level, drop=True)
        .rename_axis(outer_index_name)
        .reset_index(drop=drop_outer_index)
    )


def overwrite_dir(the_path, overwrite=True):
    """
    Overwrites a directory at a path.
    Will fail if overwrite=False and the_path exists
    """
    if os.path.exists(the_path):
        if not overwrite:
            raise ValueError(
                "Trying to overwrite directory {}, but `overwrite` is False".format(
                    the_path
                )
            )
        shutil.rmtree(the_path)
    os.makedirs(the_path)


def read_file(
    filename, columns=None, load_extension="parquet", mode="pandas", **kwargs
):

    if mode == "pandas":
        if load_extension == "parquet":
            return pd.read_parquet(filename, columns=columns, **kwargs)
        elif load_extension == "csv":
            return pd.read_csv(filename, usecols=columns, **kwargs)
    elif mode == "dask":
        if load_extension == "parquet":
            return pd.read_parquet(filename, columns=columns, **kwargs)
        elif load_extension == "csv":
            return pd.read_csv(filename, usecols=columns, **kwargs)
    else:
        raise ValueError('"pandas" and "dask" are the only allowable modes')
