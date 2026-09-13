"""CI acceptance tests: layout independence and fail-closed scientific checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from compare_results import _json_default
from compare_results import compare_results
from compatibility.normalize import scalar_metadata
from layout_fixtures import (
    test_loaders_project_monolithic_and_split_layouts_by_labels as make_pair,
)

CONTRACT = {
    "defaults": {
        "fitted_data_normalized_rms": 1e-6,
        "parameter_rtol": 1e-4,
        "parameter_atol": 1e-8,
    },
    "scenarios": [{"id": "fixture", "result": ".", "notebook": "fixture.ipynb"}],
}


@pytest.mark.parametrize("value", [np.int64(1), np.array(1), np.array([1]), np.array([[[1]]])])
def test_scalar_metadata_single_element(value):
    assert scalar_metadata(value) == 1
    assert isinstance(scalar_metadata(value), int)
    assert json.loads(json.dumps(value, default=_json_default)) == 1


@pytest.mark.parametrize("value", [np.array([]), np.array([1, 2]), np.array([[1, 2]])])
def test_scalar_metadata_non_scalar(value):
    assert scalar_metadata(value) is value
    assert json.loads(json.dumps(value, default=_json_default)) == str(value)


def report(tmp_path):
    return compare_results(tmp_path / "main", tmp_path / "staging", CONTRACT)


@pytest.mark.parametrize("layout", ["main", "staging"])
def test_same_version(tmp_path, layout):
    make_pair(tmp_path)
    result = compare_results(tmp_path / layout, tmp_path / layout, CONTRACT)
    assert result["summary"]["acceptable"]


def test_cross_version(tmp_path):
    make_pair(tmp_path)
    assert report(tmp_path)["summary"]["acceptable"]


@pytest.mark.parametrize(
    "field",
    [
        "input_data.nc",
        "fitted_data.nc",
        "residuals.nc",
        "fit_decomposition/clp.nc",
        "fit_decomposition/matrix.nc",
    ],
)
def test_missing_artifact(tmp_path, field):
    make_pair(tmp_path)
    (tmp_path / "staging/optimization_results/dataset1" / field).unlink()
    assert not report(tmp_path)["summary"]["acceptable"]


@pytest.mark.parametrize("mutation", ["input", "fit", "nan", "coordinate", "extra_coordinate"])
def test_scientific_regression(tmp_path, mutation):
    make_pair(tmp_path)
    name = "input_data.nc" if mutation == "input" else "fitted_data.nc"
    path = tmp_path / "staging/optimization_results/dataset1" / name
    ds = xr.load_dataset(path)
    var = next(iter(ds.data_vars))
    if mutation == "coordinate":
        ds = ds.assign_coords(time=[0.0, 2.0])
    elif mutation == "extra_coordinate":
        ds = ds.reindex(time=[0.0, 1.0, 2.0], fill_value=1.0)
    else:
        ds[var].values[0, 0] += 0.01
        if mutation == "nan":
            ds[var].values[0, 0] = np.nan
    ds.to_netcdf(path)
    assert not report(tmp_path)["summary"]["acceptable"]


def test_empty_contract_rejected(tmp_path):
    with pytest.raises(ValueError):
        compare_results(tmp_path, tmp_path, {"scenarios": []})


def test_missing_parameter_declaration(tmp_path):
    make_pair(tmp_path)
    path = tmp_path / "staging/result.yml"
    doc = yaml.safe_load(path.read_text())
    del doc["optimized_parameters"]
    path.write_text(yaml.safe_dump(doc))
    assert not report(tmp_path)["summary"]["acceptable"]


@pytest.mark.parametrize("change", ["parameters", "decomposition", "metadata"])
def test_secondary_differences_do_not_override_fit(tmp_path, change):
    make_pair(tmp_path)
    if change == "parameters":
        (tmp_path / "staging/optimized_parameters.csv").write_text("label,value\np,200\n")
    elif change == "decomposition":
        path = tmp_path / "staging/optimization_results/dataset1/fit_decomposition/clp.nc"
        ds = xr.load_dataset(path) * 10
        ds.to_netcdf(path)
    else:
        path = tmp_path / "staging/result.yml"
        doc = yaml.safe_load(path.read_text())
        doc["optimization_results"]["dataset1"]["meta"] = {"dataset_scale": 300}
        path.write_text(yaml.safe_dump(doc))
    assert report(tmp_path)["summary"]["acceptable"]


def test_explicit_scenario_tolerance(tmp_path):
    make_pair(tmp_path)
    path = tmp_path / "staging/optimization_results/dataset1/fitted_data.nc"
    ds = xr.load_dataset(path) * (1 + 2e-5)
    ds.to_netcdf(path)
    assert not report(tmp_path)["summary"]["acceptable"]
    contract = {
        **CONTRACT,
        "scenarios": [{**CONTRACT["scenarios"][0], "fitted_data_normalized_rms": 3e-5}],
    }
    assert compare_results(tmp_path / "main", tmp_path / "staging", contract)["summary"][
        "acceptable"
    ]
