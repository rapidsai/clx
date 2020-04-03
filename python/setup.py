from setuptools import setup, find_packages
import os
import sys

install_requires = [
    "confluent_kafka",
    "transformers",
    "seqeval[gpu]",
    "python-whois",
    "requests",
    "mockito",
    "torch==1.3.1",
    "cython"
]

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'

setup(
    name="clx",
    version="0.13.0",
    description="CLX",
    author="NVIDIA Corporation",
    packages=find_packages(include=["clx", "clx.*"]),
    package_data={
        "clx.parsers": ["resources/*.yaml"],
        "clx.dns": ["resources/*.txt"],
        "clx.heuristics": ["resources/*.csv"],
    },
    install_requires=install_requires
)
