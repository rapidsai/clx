from setuptools import setup, find_packages

install_requires = ["confluent_kafka"]

setup(
      name="rapidscyber",
      version="0.1.0",
      description="RAPIDS Cyber",
      author="NVIDIA Corporation",
      packages=find_packages(include=["rapidscyber", "rapidscyber.*"]),
      install_requires=install_requires,
)
