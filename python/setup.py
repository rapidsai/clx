import os
import sys

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import versioneer
from distutils.sysconfig import get_python_lib

INSTALL_REQUIRES = [
    "confluent_kafka",
    "transformers",
    "seqeval[gpu]",
    "python-whois",
    "requests",
    "mockito",
    "torch==1.3.1",
    "cython"
]

if int(os.environ.get('CONDA_BUILD', 0)) == 1:
    install_requires = []

conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_lib_dir = conda_prefix + '/lib'

EXTENSIONS = [
    Extension(
        "*",
        sources=["clx/analytics/tokenizer_wrapper.pyx"],
        language="c++",
        extra_compile_args=['-std=c++14'],
        runtime_library_dirs=[conda_lib_dir],
        library_dirs=[get_python_lib()],
        libraries=["clx"]
    )
]

setup(
    name="clx",
    version=versioneer.get_version(),
    description="CLX",
    author="NVIDIA Corporation",
    setup_requires=['cython'],
    ext_modules=cythonize(EXTENSIONS),
    packages=find_packages(include=["clx", "clx.*"]),
    package_data={
        "clx.analytics": ["resources/*.txt"],
        "clx.parsers": ["resources/*.yaml"],
        "clx.dns": ["resources/*.txt"],
        "clx.heuristics": ["resources/*.csv"]
    },
    install_requires=INSTALL_REQUIRES,
    license="Apache",
    cmdclass=versioneer.get_cmdclass()
)
