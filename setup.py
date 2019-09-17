from setuptools import setup, find_packages

install_requires = ["confluent_kafka"]

setup(
      name="clx",
      version="0.10.0",
      description="CLX",
      author="NVIDIA Corporation",
      packages=find_packages(include=["clx", "clx.*"]),
      install_requires=install_requires,
)
