import os
import sys

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

from distutils.sysconfig import get_python_lib

install_requires = ["confluent_kafka", "transformers", "seqeval[gpu]", "python-whois", "requests", "mockito", "torch==1.3.1", "cython"]

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'

EXTENSIONS = [
    Extension("*",
        sources=["clx/tokenizers/tokenizer.pyx"],
        language="c++",
        runtime_library_dirs=[conda_lib_dir],
        library_dirs=[get_python_lib()],
        libraries=["tokenizers"]
    )
]

setup(
    name="clx",
    version="0.13.0",
    description="CLX",
    author="NVIDIA Corporation",
    setup_requires=['cython'],
    ext_modules=cythonize(EXTENSIONS),
    packages=find_packages(include=["clx", "clx.*"]),
    package_data={
        "clx.parsers": ["resources/*.yaml"],
        "clx.dns": ["resources/*.txt"],
        "clx.heuristics": ["resources/*.csv"],
    },
    install_requires=install_requires
)
