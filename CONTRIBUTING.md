# Contributing to CLX

If you are interested in contributing to CLX, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/clx/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/rapidsai/clx/blob/main/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/clx/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/clx/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Fork the CLX repo and Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/clx/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/clx/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

### Build from Source

The following instructions are for developers and contributors to CLX OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build CLX from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The CLX package include both a C/C++ CUDA portion and a python portion.  Both libraries need to be installed in order CLX to operate correctly.

The following instructions are tested on Linux systems.

#### Prerequisites

Compiler requirement:

* `gcc`     version 5.4+
* `nvcc`    version 10.0+
* `cmake`   version 3.12

CUDA requirement:

* CUDA 10.0+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

Since `cmake` will download and build Apache Arrow you may need to install Boost C++ (version 1.58+) before running
`cmake`:

#### Build and Install the C/C++ CUDA components

To install CLX from source, ensure the dependencies are met and follow the steps below:

1) Clone the repository and submodules

  ```bash
  # Set the location to CLX in an environment variable CLX_HOME
  export CLX_HOME=$(pwd)/clx

  # Download the CLX repo
  git clone https://github.com/rapidsai/clx.git $CLX_HOME


2) Create the conda development environment

```bash
# create the conda environment (assuming in base `clx` directory)

conda env create --name clx_dev --file conda/environments/clx_dev.yml

# activate the environment
conda activate clx_dev

# to deactivate an environment
conda deactivate
```

  - The environment can be updated as development includes/changes the dependencies. To do so, run:


```bash

conda env update --name clx_dev --file conda/environments/clx_dev.yml

conda activate clx_dev
```

3) Build and install `libclx`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

  This project uses cmake for building the C/C++ library.

  ```bash
  # Set the location to CLX in an environment variable CLX_HOME
  export CLX_HOME=$(pwd)/clx

  cd $CLX_HOME
  cd cpp                                        # enter cpp directory
  mkdir build                                   # create build directory
  cd build                                      # enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

  # now build the code
  make -j                                       # "-j" starts multiple threads
  make install                                  # install the libraries
  ```

As a convenience, a `build.sh` script is provided in `$CLX_HOME`. To execute the same build commands above, run the script as shown below.  Note that the libraries will be installed to the location set in `$PREFIX` if set (i.e. `export PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ cd $CLX_HOME
$ ./build.sh libclx  # build the CLX libraries and install them to
                         # $PREFIX if set, otherwise $CONDA_PREFIX
```

#### Building and installing the Python package

5. Install the Python package to your Python path:

```bash
cd $CLX_HOME
cd python
python setup.py build_ext --inplace
python setup.py install    # install CLX python bindings
```

Like the `libclx` build step above, `build.sh` can also be used to build the `clx` python package, as shown below:
```bash
$ cd $CLX_HOME
$ ./build.sh clx  # build the clx python bindings and install them
                      # to $PREFIX if set, otherwise $CONDA_PREFIX
```

Note: other `build.sh` options include:
```bash
$ cd $CLX_HOME
$ ./build.sh clean                        # remove any prior build artifacts and configuration (start over)
$ ./build.sh libclx -v                # compile and install libclx with verbose output
$ ./build.sh libclx -g                # compile and install libclx for debug
$ PARALLEL_LEVEL=4 ./build.sh libclx  # compile and install libclx limiting parallel build jobs to 4 (make -j4)
$ ./build.sh libclx -n                # compile libclx but do not install
```


Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

## Creating documentation

Python API documentation can be generated from [docs](docs) directory.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md