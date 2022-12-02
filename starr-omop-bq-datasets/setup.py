"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import find_packages, setup

setup(
    name="datasets",
    version="0.0.0",
    description="STARR OMOP cohorts and labelers",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
    ],
)
