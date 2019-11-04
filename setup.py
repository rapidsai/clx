from setuptools import setup, find_packages

install_requires = ["confluent_kafka", "python-whois", "requests", "torch"]

setup(
      name="clx",
      version="0.11.0",
      description="CLX",
      author="NVIDIA Corporation",
      packages=find_packages(include=["clx", "clx.*"]),
      package_data={
            "clx.parsers":["resources/*.yaml"],
            "clx.dns":["resources/*.txt"],
            "clx.heuristics":["resources/*.csv"]
      },
      install_requires=install_requires
)