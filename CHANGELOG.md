# clx 23.02.00 (Date TBD)

Please see https://github.com/rapidsai/clx/releases/tag/v23.02.00a for the latest changes to this development branch.

# clx 22.12.00 (Date TBD)

Please see https://github.com/rapidsai/clx/releases/tag/v22.12.00a for the latest changes to this development branch.

# clx 22.10.00 (12 Oct 2022)


# clx 22.08.00 (17 Aug 2022)

## 🐛 Bug Fixes

- Fix style and invalid regex pattern ([#503](https://github.com/rapidsai/clx/pull/503)) [@bsuryadevara](https://github.com/bsuryadevara)

## 📖 Documentation

- Switch to using common `js` &amp; `css` code ([#507](https://github.com/rapidsai/clx/pull/507)) [@galipremsagar](https://github.com/galipremsagar)

# clx 22.06.00 (7 Jun 2022)

## 🐛 Bug Fixes

- cyBERT and WEP fixes after upstream cuDF updates ([#487](https://github.com/rapidsai/clx/pull/487)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add transformers dependency and update notebooks ([#486](https://github.com/rapidsai/clx/pull/486)) [@bsuryadevara](https://github.com/bsuryadevara)
- Update cybert_example_training.ipynb ([#469](https://github.com/rapidsai/clx/pull/469)) [@raykallen](https://github.com/raykallen)

## 📖 Documentation

- doc updates ([#489](https://github.com/rapidsai/clx/pull/489)) [@bsuryadevara](https://github.com/bsuryadevara)

## 🛠️ Improvements

- Simplify conda recipe ([#493](https://github.com/rapidsai/clx/pull/493)) [@Ethyling](https://github.com/Ethyling)
- Use conda to build python packages during GPU tests ([#484](https://github.com/rapidsai/clx/pull/484)) [@Ethyling](https://github.com/Ethyling)
- Remove `splunklib` directories ([#483](https://github.com/rapidsai/clx/pull/483)) [@efajardo-nv](https://github.com/efajardo-nv)
- Use conda compilers ([#476](https://github.com/rapidsai/clx/pull/476)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#472](https://github.com/rapidsai/clx/pull/472)) [@Ethyling](https://github.com/Ethyling)
- Support Loda save and load operations. ([#448](https://github.com/rapidsai/clx/pull/448)) [@tzemicheal](https://github.com/tzemicheal)
- Edit variable formatting for API document page ([#386](https://github.com/rapidsai/clx/pull/386)) [@msadang](https://github.com/msadang)

# clx 22.04.00 (6 Apr 2022)

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality ([#478](https://github.com/rapidsai/clx/pull/478)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `.github/ops-bot.yaml` config file ([#477](https://github.com/rapidsai/clx/pull/477)) [@ajschmidt8](https://github.com/ajschmidt8)
- removed deprecated function to_array ([#474](https://github.com/rapidsai/clx/pull/474)) [@bsuryadevara](https://github.com/bsuryadevara)

# clx 22.02.00 (2 Feb 2022)


# clx 21.12.00 (9 Dec 2021)

## 🐛 Bug Fixes

- Update sequence_classifier.py ([#459](https://github.com/rapidsai/clx/pull/459)) [@raykallen](https://github.com/raykallen)
- Update CLX workflow yaml reader for pyyaml v6.0 ([#458](https://github.com/rapidsai/clx/pull/458)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🛠️ Improvements

- Fix Changelog Merge Conflicts for `branch-21.12` ([#463](https://github.com/rapidsai/clx/pull/463)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update to PyTorch 1.10 ([#460](https://github.com/rapidsai/clx/pull/460)) [@efajardo-nv](https://github.com/efajardo-nv)

# clx 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- Fix binary sequence classifier tokenizer initialization ([#452](https://github.com/rapidsai/clx/pull/452)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🚀 New Features

- DGA extended sequence handling ([#449](https://github.com/rapidsai/clx/pull/449)) [@bsuryadevara](https://github.com/bsuryadevara)

## 🛠️ Improvements

- Fix Forward-Merge Conflicts ([#450](https://github.com/rapidsai/clx/pull/450)) [@ajschmidt8](https://github.com/ajschmidt8)
- support space in workspace ([#383](https://github.com/rapidsai/clx/pull/383)) [@jolorunyomi](https://github.com/jolorunyomi)

# clx 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Fix filter function in periodicity detection ([#440](https://github.com/rapidsai/clx/pull/440)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update conda development environment ([#428](https://github.com/rapidsai/clx/pull/428)) [@efajardo-nv](https://github.com/efajardo-nv)

## 📖 Documentation

- Update README getting started code ([#433](https://github.com/rapidsai/clx/pull/433)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🛠️ Improvements

- variable fix so that accuracy_score doesn&#39;t throw an error ([#438](https://github.com/rapidsai/clx/pull/438)) [@taureandyernv](https://github.com/taureandyernv)
- Update `conda` environment name for CI ([#436](https://github.com/rapidsai/clx/pull/436)) [@ajschmidt8](https://github.com/ajschmidt8)
- Drop `flatbuffers` install ([#434](https://github.com/rapidsai/clx/pull/434)) [@jakirkham](https://github.com/jakirkham)
- Update README instructions to include stable version ([#432](https://github.com/rapidsai/clx/pull/432)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update notebooks to use the new Tokenizer ([#430](https://github.com/rapidsai/clx/pull/430)) [@VibhuJawa](https://github.com/VibhuJawa)
- Update sphinx config ([#427](https://github.com/rapidsai/clx/pull/427)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix syntax highlighting in README ([#425](https://github.com/rapidsai/clx/pull/425)) [@garethsb](https://github.com/garethsb)
- Fix `21.08` forward-merge conflicts ([#424](https://github.com/rapidsai/clx/pull/424)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use `cupy.ndarray` in docstrings ([#410](https://github.com/rapidsai/clx/pull/410)) [@jakirkham](https://github.com/jakirkham)
- Fix merge conflicts in #408 ([#409](https://github.com/rapidsai/clx/pull/409)) [@ajschmidt8](https://github.com/ajschmidt8)
- Added custreamz and cugraph integration demo ([#405](https://github.com/rapidsai/clx/pull/405)) [@shaneding](https://github.com/shaneding)
- Fix merge conflicts ([#404](https://github.com/rapidsai/clx/pull/404)) [@ajschmidt8](https://github.com/ajschmidt8)
- change pretrained model ([#387](https://github.com/rapidsai/clx/pull/387)) [@gbatmaz](https://github.com/gbatmaz)

# clx 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Update cuml imports ([#407](https://github.com/rapidsai/clx/pull/407)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🚀 New Features

- REVIEW: ABP notebook Training+Inference ([#398](https://github.com/rapidsai/clx/pull/398)) [@gbatmaz](https://github.com/gbatmaz)

## 🛠️ Improvements

- Update conda dev environments ([#406](https://github.com/rapidsai/clx/pull/406)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update `CHANGELOG.md` links for calver ([#402](https://github.com/rapidsai/clx/pull/402)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update docs build script ([#401](https://github.com/rapidsai/clx/pull/401)) [@ajschmidt8](https://github.com/ajschmidt8)

# clx 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Update Network Mapping notebook to skip Louvain on Pascal ([#362](https://github.com//rapidsai/clx/pull/362)) [@efajardo-nv](https://github.com/efajardo-nv)
- Network mapping notebook fix ([#360](https://github.com//rapidsai/clx/pull/360)) [@efajardo-nv](https://github.com/efajardo-nv)

## 📖 Documentation

- Readme updates ([#373](https://github.com//rapidsai/clx/pull/373)) [@msadang](https://github.com/msadang)

## 🚀 New Features

- add pii training notebook ([#375](https://github.com//rapidsai/clx/pull/375)) [@raykallen](https://github.com/raykallen)
- Add IDS (malware detection) usecase example using Loda. ([#369](https://github.com//rapidsai/clx/pull/369)) [@tzemicheal](https://github.com/tzemicheal)
- anomaly detection dbscan ([#357](https://github.com//rapidsai/clx/pull/357)) [@brhodes10](https://github.com/brhodes10)
- Dataset binary and frequency features ([#347](https://github.com//rapidsai/clx/pull/347)) [@brhodes10](https://github.com/brhodes10)

## 🛠️ Improvements

- Update Changelog Link ([#371](https://github.com//rapidsai/clx/pull/371)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update sequence classifier data loader ([#370](https://github.com//rapidsai/clx/pull/370)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add checkpoint support to sequence classifier ([#368](https://github.com//rapidsai/clx/pull/368)) [@efajardo-nv](https://github.com/efajardo-nv)
- Prepare Changelog for Automation ([#367](https://github.com//rapidsai/clx/pull/367)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove deprecated phishing detection module ([#366](https://github.com//rapidsai/clx/pull/366)) [@efajardo-nv](https://github.com/efajardo-nv)
- Branch 0.19 merge 0.18 ([#365](https://github.com//rapidsai/clx/pull/365)) [@efajardo-nv](https://github.com/efajardo-nv)
- Documentation fixes for EDA and Features modules ([#364](https://github.com//rapidsai/clx/pull/364)) [@brhodes10](https://github.com/brhodes10)
- Fix merge conflicts in #339 ([#340](https://github.com//rapidsai/clx/pull/340)) [@ajschmidt8](https://github.com/ajschmidt8)

# clx 0.18.0 (24 Feb 2021)

## Bug Fixes 🐛

- Fix Kafka download in CLX Streamz Docker image (#349) @efajardo-nv
- Fix PageRank in Network Mapping notebook (#342) @efajardo-nv
- Port heuristic update (#336) @efajardo-nv
- Fix pytorch version in conda recipe (#321) @efajardo-nv
- Pin to PyTorch 1.7.0 (#320) @efajardo-nv

## Documentation 📖

- OSI Docs (#331) @bsuryadevara
- README updates (#326) @efajardo-nv

## New Features 🚀

- CLX Module for LODA Anomaly Detection (#335) @efajardo-nv
- SlashNext API Integration (#328) @bsuryadevara
- Loda algorithm for anomaly detection and explanation. (#325) @tzemicheal
- Exploratory Data Analysis Summary Stats (#324) @brhodes10
- Auto-label PRs based on their content (#318) @jolorunyomi
- Add xgboost training example (#313) @raykallen

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#353) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#351) @Ethyling
- Update to PyTorch 1.7.1 (#343) @efajardo-nv
- Prepare Changelog for Automation (#337) @ajschmidt8
- Add PyTorch DataParallel to cyBERT (#327) @efajardo-nv
- Update to HuggingFace Transformers v4 (#322) @efajardo-nv
- Adding Predictive Maintenance &amp; Root Cause Analysis Notebook (#317) @gbatmaz
- Fix DGA Detection Notebook and DetectorDataset Memory Errors (#315) @bsuryadevara
- Update CLX Streamz workflow to simplify new workflow implementation (#308) @bsuryadevara
- Update to FIL Streamz example notebook (#307) @brhodes10
- CLX sequence classifier (#303) @efajardo-nv
- gpuCI Scripts Update (#276) @dillon-cullinan

# clx 0.17.0 (10 Dec 2020)

## New Features
- PR #277 cyBERT class to support DistilBERT and ELECTRA
- PR #273 FIL streamz notebook
- PR #281 CLX module for periodicity detection

## Improvements
- PR #248 Fix to cybert streamz to handle larger output messages
- PR #263 Pin cmake policies to cmake 3.17 version
- PR #272 CLX Streamz generic structure
- PR #283 Remove CLX subword tokenizer
- PR #286 Update to PyTorch 1.7 for CUDA 11 support
- PR #287 Dockerfile updates
- PR #292 Phishing detection updates and workflow
- PR #288 Update network mapping notebooks to download from S3
- PR #296 cyBERT load model function update

## Bug Fixes
- PR #269 Add cybert class to docs build
- PR #271 Alert Analysis notebook fix
- PR #291 DNS Extractor fix
- PR #294 API doc fixes
- PR #295 Docs build script fix
- PR #298 Zeek parser fix

# clx 0.16.0 (21 Oct 2020)

## New Features
- PR #208 CLX Supervised Asset Classification
- PR #265 Add FFT notebook
## Improvements
- PR #233 Removed s3 dependency from dga detection unit-test
- PR #235 Pre-commit hooks for isort, black and flake8
- PR #232 Update phishing detection to use cudf subword tokenizer
- PR #237 Cybert streamz memory optimization
- PR #238 Deprecate CLX subword tokenizer
- PR #239 Added cybert streamz log size parameter
- PR #244 Add cybert dataloader
- PR #247 Allow CuPy 8.x
- PR #246 Dockerfile updates
- PR #262 CLX Streamz Getting Started
- PR #253 Save clx dga model as dictionary

## Bug Fixes
- PR #231 Fix segmentation fault in cybert notebook
- PR #229 Fix cybert streamz script
- PR #236 Fix RAPIDS logo in READMEs
- PR #240 Fix errors from upstream changes to cuDF DataFrame.drop and RMM
- PR #241 Fix unit tests after cuDF update
- PR #242 Pin seqeval to v0.0.12 for cybert example training notebook
- PR #243 Fix error handling in `ci/gpu/build.sh`
- PR #256 Fix missing projects variable from docs build
- PR #261 Fix incorrect workspace variable in docs build

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
