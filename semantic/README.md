# Version-aware semantic example comparison

This payload is exported from the validation workspace's `validation/` adapter.
It runs without importing pyglotaran. `data` in result.yml selects v0.7;
`optimization_results` selects v0.8, independently for reference and current.

Run `python semantic/compare_results.py --main-root comparison-results
--staging-root comparison-results-current --output semantic-report.json` (one
line), or select `comparison_mode: semantic` in the composite action.
`semantic/scenarios.yml` is the explicit coverage/tolerance contract.

Inputs and dimension labels compare exactly. Fitted-data normalized RMS is the
scientific gate. Secondary parameter/decomposition/metadata differences are
reported; missing required or declared files, invalid layouts, non-finite input
or fit values, coordinate mismatches and empty contracts fail. Labels may reorder
and named dimensions may transpose, but extra dimension labels cannot disappear.

Test with `python -m pytest semantic/tests -q`. The 2e-6 spectral-guidance
exception accepts observed constrained-fit optimizer-path drift 1.257246e-6;
it is not proof of convergence. Default is 1e-6, transient two-dataset 2e-5,
weighted 3D 3e-5. Do not increase these without fresh paired evidence.

Deployment requires publishing this action and updating the consumer validation
gitlink. Existing validator commit ae5af096a186833c181871f7b0435d40858e3a98
does not include this payload. The legacy historical baseline must be refreshed
for the current 14-leaf contract before enabling semantic mode in remote CI.
