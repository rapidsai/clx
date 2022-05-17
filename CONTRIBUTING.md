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

### Code Formatting

#### Python

CLX uses [Black](https://black.readthedocs.io/en/stable/),
[isort](https://readthedocs.org/projects/isort/), and
[flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent code format
throughout the project. `Black`, `isort`, and `flake8` can be installed with
`conda` or `pip`:

```bash
conda install black isort flake8
```

```bash
pip install black isort flake8
```

These tools are used to auto-format the Python code, as well as check the Cython
code in the repository. Additionally, there is a CI check in place to enforce
that committed code follows our standards. You can use the tools to
automatically format your python code by running:

```bash
isort --atomic python/**/*.py
black python
```

and then check the syntax of your Python code by running:

```bash
flake8 python
```

Additionally, many editors have plugins that will apply `isort` and `Black` as
you edit files, as well as use `flake8` to report any style / syntax issues.

#### Pre-commit hooks

Optionally, you may wish to setup [pre-commit hooks](https://pre-commit.com/)
to automatically run `isort`, `Black`, and `flake8` when you make a git commit.
This can be done by installing `pre-commit` via `conda` or `pip`:

```bash
conda install -c conda-forge pre_commit
```

```bash
pip install pre-commit
```

and then running:

```bash
pre-commit install
```

from the root of the CLX repository. Now `isort`, `Black`, and `flake8` will be
run each time you commit changes.

## Script to build CLX from source

### Build from Source

The following instructions are for developers and contributors to CLX OSS development. These instructions are tested on Linux Ubuntu 18.04 & 20.04. Use these instructions to build CLX from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The following instructions are tested on Linux systems.

#### Prerequisites

CUDA requirement:

* CUDA 11.5
* NVIDIA driver 470.82+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

To install CLX from source, ensure the dependencies are met and follow the steps below:

Clone the repository and submodules:

```bash
  # Set the location to CLX in an environment variable CLX_HOME
  export CLX_HOME=$(pwd)/clx

  # Download the CLX repo
  git clone https://github.com/rapidsai/clx.git $CLX_HOME
```

Create the conda development environment:

```bash
# create the conda environment (assuming in base `clx` directory)

mamba env create --name clx_dev --file conda/environments/clx_dev_cuda11.5.yml

# activate the environment
conda activate clx_dev

# to deactivate an environment
conda deactivate
```

The environment can be updated as development includes/changes the dependencies. To do so, run:

```bash
mamba env update --name clx_dev --file conda/environments/clx_dev_cuda11.5.yml

conda activate clx_dev
```

Build the `clx` python package:

```bash
$ cd $CLX_HOME/python
$ python setup.py install
```

## Creating documentation

Python API documentation can be generated from [docs](docs) directory.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
