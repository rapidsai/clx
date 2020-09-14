# clx 0.16.0 (Date TBD)

## New Features
- PR #208 CLX Supervised Asset Classification

## Improvements
- PR #233 Removed s3 dependency from dga detection unit-test
- PR #235 Pre-commit hooks for isort, black and flake8
- PR #232 Update phishing detection to use cudf subword tokenizer
- PR #237 Cybert streamz memory optimization

## Bug Fixes
- PR #231 Fix segmentation fault in cybert notebook
- PR #229 Fix cybert streamz script
- PR #236 Fix RAPIDS logo in READMEs

# clx 0.15.0 (26 Aug 2020)

## New Features
- PR #189 Add Cybert class
- PR #170 Add CLX Code of Conduct
- PR #176 Add Periodicity Detection Notebook
- PR #193 AC notebook

## Improvements
- PR #202 Update cybert notebook to use cudf tokenizer
- PR #174 Install dependencies via meta package
- PR #175 GPU tokenizer updates
- PR #181 CLX Query to support dask blazingsql
- PR #159 Update to PyTorch 1.5
- PR #191 Update conda upload versions for new supported CUDA/Python
- PR #203 CLX docker and conda environment updates
- PR #209 GPU build and Dockerfile update
- PR #211 PyTorch 1.6 support
- PR #214 Rename BERT vocab and hash file names for disambiguation
- PR #215 Update dependencies in gpu build, conda recipe, and dockerfile
- PR #220 README update

## Bug Fixes
- PR #169 Fix documentation links
- PR #171 Fix errors from nvstrings removal
- PR #178 Update local build to use new gpuCI image
- PR #180 Remove remaining references to nvstrings
- PR #182 Fix errors from update to cudf's column rename method
- PR #185 Update gpu build to use latest dask
- PR #190 Fix issue with incorrect docker image being used in local build script
- PR #192 Conda build fix
- PR #197 Fix errors from Python 3.8 conda build and cudf update
- P# #205 Build and notebook fixes/updates

# clx 0.14.0 (Date TBD)

## New Features
- PR #140 Added cybert class and streamz workflow
- PR #141 CUDA BERT Tokenizer
- PR #152 Local gpuCI build script
- PR #133 Phishing detection using BERT

## Improvements
- PR #149 Add Versioneer
- PR #151 README and CONTRIBUTING updates
- PR #160 Build script updates
- PR #167 Add git commit to conda package
- PR #172 Add docs build script
- PR #155 Tokenizer rmm integration
- PR #172 Add docs build script

## Bug Fixes
- PR #150 Fix splunk alert workflow test
- PR #154 Local gpuCI build fix
- PR #158 Fix test cases to support latest cudf changes
- PR #164 GPU build fix

# clx 0.13.0 (Date TBD)

## New Features
- PR #130 Example of streamz
- PR #132 CLX query applications

## Improvements
- PR #103 DGA detector refactor
- PR #120 Use pytest tmpdir fixtures in unit tests
- PR #125 Added notebook testing to gpuCI gpu build
- RR #144 Python refactor

## Bug Fixes
- PR #123 Fix update-version.sh
- PR #129 Fix alert analysis notebook viz using cuxfilter
- PR #138 Fix test cases to support latest cudf changes

# clx 0.12.0 (Date TBD)

## New Features
- PR #93 Add Bokeh visualization back to Alert Analysis notebook

## Improvements
- PR #88 Documentation updates
- PR #85 Add codeowners
- PR #86 Add issue templates
- PR #87 CLX docker updates
- PR #95 Download Alert Analysis data from S3
- PR #101 Refactor DNS & IP code
- PR #108 JSON read/write support
- PR #105 Documentation updates to README

## Bug Fixes
- PR #97 Notebook cleanup and gitignore update
- PR #102 - Fix error from renamed cuxfilter module
- PR #107 Fixes to workflow notebook
- PR #109 Fix to cybert notebook
- PR #117 Fix to dga detector str2ascii

# clx 0.11.0 (Date TBD)

## New Features
 - PR #74 Updated cyBERT notebook
 - PR #66 CLX Read The Docs
 - PR #64 Added cybert notebook and data
 - PR #54 Added Network Mapping notebook
 - PR #48 Added port heuristic to detect major ports
 - PR #60 Added DGA detection notebook and DNS log parsing notebook
 - PR #76 Added update-version script

## Improvements
 - PR #70 Sphinx doc formatting improvements
 - PR #58 Update docker image
 - PR #55 Updates to folder structure
 - PR #52 Include DNS and OSI usage to notebook.
 - PR #49 Parameter pass-through to underlying cudf/dask_cudf
 - PR #47 Update splunk workflow output
 - PR #43 Functionality to parse selected windows events
 - PR #46 Adding copyright
 - PR #53 Autogenerate api docs
 - PR #71 Remove unused CUDA conda labels

## Bug Fixes
 - PR #81 Reader filepath fix
 - PR #77 Fix to unit test
 - PR #69 Simple fix to DNS
 - PR #68 Update to Alert Analysis Workflow notebooks
 - PR #67 Fix DNS extra columns
 - PR #50 Workflow IO fix
 - PR #45 More Kafka IO fixes
 - PR #44 Fix Kafka IO
 - PR #42 Include DNS parser in module
 - PR #41 Fix unit test
 - PR #39 Fix gpuCI builds

# clx 0.10.0 (Date TBD)

## New Features
 - PR #59 Added clx workflow implementation notebooks
 - PR #48 Added port heuristic to detect major ports
 - PR #35 Added readthedocs
 - PR #37 Add pytorch dependency.
 - PR #37 Add DGA detection feature.
 - PR #14 Integrate repo with gpuCI

## Improvements

## Bug Fixes
