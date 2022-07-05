import versioneer
from setuptools import find_packages, setup

setup(name="clx",
      version=versioneer.get_version(),
      description="CLX",
      author="NVIDIA Corporation",
      packages=find_packages(include=["clx", "clx.*"]),
      package_data={
          "clx.analytics": ["resources/*.txt"],
          "clx.parsers": ["resources/*.yaml"],
          "clx.dns": ["resources/*.txt"],
          "clx.heuristics": ["resources/*.csv"]
      },
      license="Apache",
      cmdclass=versioneer.get_cmdclass())
