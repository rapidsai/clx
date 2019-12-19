# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;Cyber Log Accelerators (CLX)</div>

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/clx/blob/master/README.md) ensure you are on the `master` branch.

CLX ("clicks") provides a collection of [RAPIDS](https://rapids.ai/) examples for security analysts, data scientists, and engineers to quickly get started applying RAPIDS and GPU acceleration to real-world cybersecurity use cases.

The goal of CLX is to:

1. Provide SIEM integration with GPU compute environments via RAPIDS and effectively extend the SIEM environment,
1. Make available pre-built use cases that demonstrate CLX and RAPIDS functionality that are ready to use in a Security Operations Center (SOC), 
1. Allow cyber data scientists and SecOps teams to generate workflows, using cyber-specific GPU-accelerated primitives and methods, that let them interact with code using security language, and
1. Accelerate log parsing in a flexible, non-regex method.


## Getting Started

CLX is targeted towards cybersecurity data scientists, senior security analysts, threat hunters, and forensic investigators. Data scientists can use CLX in traditional Python files and Jupyter notebooks. CLX also includes structure in the form of a workflow. A workflow is a series of data transformations performed on a [GPU dataframe](https://github.com/rapidsai/cudf) that contains raw cyber data, with the goal of surfacing meaningful cyber analytical output. Multiple I/O methods are available, including Kafka and on-disk file stores.

Example flow workflow reading and writing to file:

```python
from clx.workflow import netflow_workflow

source = {
   "type": "fs",
   "input_format": "csv",
   "input_path": "/path/to/input",
   "schema": ["firstname","lastname","gender"],
   "delimiter": ",",
   "required_cols": ["firstname","lastname","gender"],
   "dtype": ["str","str","str"],
   "header": "0"
}
dest = {
   "type": "fs",
   "output_format": "csv",
   "output_path": "/path/to/output"
}
wf = netflow_workflow.NetflowWorkflow(source=source, destination=dest, name="my-netflow-workflow")
wf.run_workflow()
```

## Example Notebooks
The notebooks folder contains example use cases and workflow instantiations.

## Installation
CLX is available in a Docker container, by building from source, and through Conda installation.

### CLX Docker Container

Prerequisites
* NVIDIA Pascal™ GPU architecture or better
* CUDA 9.2 or 10.0 compatible NVIDIA driver
* Ubuntu 16.04/18.04 or CentOS 7
* Docker CE v18+
* nvidia-docker v2+

Pull the RAPIDS container and build CLX image.

```aidl
docker pull rapidsai/rapidsai-dev-nightly:0.11-cuda10.0-devel-ubuntu18.04-py3.7
docker build -t clx:latest .
```

Start Container and Notebook Server

#### Preferred - Docker CE v19+ and nvidia-container-toolkit
```aidl
docker run  --gpus '"device=0"' \
  --rm -d \
  -p 8888:8888 \
  -p 8787:8787 \
  -p 8686:8686 \
  clx:latest
```

#### Legacy - Docker CE v18 and nvidia-docker2
```aidl
docker run --runtime=nvidia \
  --rm -d \
  -p 8888:8888 \
  -p 8787:8787 \
  -p 8686:8686 \
  clx:latest
```

### CLX Multi-Container for Data Ingest and SIEM Integration

Start containers for CLX, Kafka, and Zookeeper.

```aidl
docker-compose up
```

### Install from Source
You can install CLX from source on an existing RAPIDS container. A RAPIDS image suitable for your environment can be pulled from https://hub.docker.com/r/rapidsai/rapidsai/. 

```aidl
# Run tests
pip install pytest
pytest

# Build and install
python setup.py install
```
### Conda install
You can conda install CLX on an existing RAPIDS container. A RAPIDS image suitable for your environment can be pulled from https://hub.docker.com/r/rapidsai/rapidsai/. 
```
conda install -c rapidsai-nightly -c rapidsai -c nvidia -c pytorch -c conda-forge -c defaults clx
```

## Contributing

For contributing guildelines please reference our [guide for contributing](https://github.com/rapidsai/clx/blob/master/CONTRIBUTING.md).
