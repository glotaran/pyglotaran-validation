# pyglotaran-validation
This repository holds scripts for cross-validation of pyglotaran against other software

## Repository Structure
The root of this repository contains a number of folders, one for each software, framework or test suite we validate against.

The primary test suite is contained in [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples), which tests pyglotaran against itself using a set of examples and case studies that double as integration tests. This test is run as a github action each time a PR is targeting pyglotaran/main and reports back how the results from running the examples compares to the last defined.

Other software we (partially) validate against, manually, on-demand or automated are:
- paramGUI ([data](https://github.com/glotaran/pyglotaran-validation-data-paramGUI))
- TIMP ([data](https://github.com/glotaran/pyglotaran-validation-data-TIMP))
- TIM ([data](https://github.com/glotaran/pyglotaran-validation-data-TIM))

(^ these will be added in the future)
