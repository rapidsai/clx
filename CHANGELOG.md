# clx 0.13.0 (31 Mar 2020)

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


# clx 0.12.0 (04 Feb 2020)

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


# clx 0.11.0 (11 Dec 2019)

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
