from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from compatibility.load_v07 import load_v07_result
from compatibility.load_v08 import load_v08_result
from compatibility.metrics import compare_arrays


def _write_array(path: Path, array: xr.DataArray, name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.to_dataset(name=name or array.name or "value").to_netcdf(path)


def _parameters(path: Path) -> None:
    pd.DataFrame({"label": ["p"], "value": [1.0]}).to_csv(path, index=False)


def test_loaders_project_monolithic_and_split_layouts_by_labels(tmp_path: Path) -> None:
    time = np.array([0.0, 1.0])
    spectral = np.array([400.0, 500.0])
    values = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("time", "spectral"),
        coords={"time": time, "spectral": spectral},
        name="data",
    )
    clp = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("spectral", "clp_label"),
        coords={"spectral": spectral, "clp_label": ["s1", "s2"]},
        name="clp",
    )
    main = tmp_path / "main"
    main.mkdir()
    _write_array(main / "dataset1.nc", values)
    _write_array(main / "dataset1.nc", values, "residual")
    dataset = xr.Dataset(
        {"data": values, "residual": values, "fitted_data": values, "clp": clp, "matrix": clp}
    )
    dataset.to_netcdf(main / "dataset1.nc")
    _parameters(main / "optimized_parameters.csv")
    (main / "scheme.yml").write_text("{}\n", encoding="utf-8")
    (main / "result.yml").write_text(
        yaml.safe_dump(
            {
                "data": {"dataset1": "dataset1.nc"},
                "optimized_parameters": "optimized_parameters.csv",
                "scheme": "scheme.yml",
            }
        ),
        encoding="utf-8",
    )

    staging = tmp_path / "staging"
    dataset_root = staging / "optimization_results" / "dataset1"
    _write_array(dataset_root / "input_data.nc", values)
    _write_array(dataset_root / "residuals.nc", values, "residual")
    _write_array(dataset_root / "fitted_data.nc", values, "__xarray_dataarray_variable__")
    split_clp = clp.rename({"clp_label": "amplitude_label"}).transpose(
        "amplitude_label", "spectral"
    )
    _write_array(
        dataset_root / "fit_decomposition" / "clp.nc", split_clp, "__xarray_dataarray_variable__"
    )
    _write_array(
        dataset_root / "fit_decomposition" / "matrix.nc",
        split_clp,
        "__xarray_dataarray_variable__",
    )
    _parameters(staging / "optimized_parameters.csv")
    (staging / "scheme.yml").write_text("{}\n", encoding="utf-8")
    (staging / "result.yml").write_text(
        yaml.safe_dump(
            {
                "optimization_results": {
                    "dataset1": {
                        "input_data": "input_data.nc",
                        "residuals": "residuals.nc",
                        "fitted_data": "fitted_data.nc",
                        "fit_decomposition": {"clp": "clp.nc", "matrix": "matrix.nc"},
                        "meta": {},
                    }
                },
                "optimized_parameters": "optimized_parameters.csv",
                "scheme": "scheme.yml",
            }
        ),
        encoding="utf-8",
    )

    expected = load_v07_result(main, "fixture")
    current = load_v08_result(staging, "fixture")
    assert (
        compare_arrays(
            expected.datasets["dataset1"].variables["data"],
            current.datasets["dataset1"].variables["data"],
            rtol=0,
            atol=0,
        )["status"]
        == "pass"
    )
    assert (
        compare_arrays(
            expected.datasets["dataset1"].variables["clp"],
            current.datasets["dataset1"].variables["clp"],
            rtol=0,
            atol=0,
        )["status"]
        == "pass"
    )
    assert Path(current.datasets["dataset1"].source_files["clp"]).name == "clp.nc"
