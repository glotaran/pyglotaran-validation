"""Loader for the pinned v0.7 monolithic result layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from .normalize import canonicalize_array
from .normalize import scalar_metadata
from .schema import DatasetView
from .schema import ResultView

SEMANTIC_VARIABLES = {
    "data": "data",
    "residual": "residual",
    "fitted_data": "fitted_data",
    "clp": "clp",
    "matrix": "matrix",
    "weight": "weight",
    "weighted_residual": "weighted_residual",
}


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_v07_result(root: str | Path, scenario: str | None = None) -> ResultView:
    root = Path(root)
    document = _yaml(root / "result.yml")
    datasets: dict[str, DatasetView] = {}
    data_mapping: dict[str, str] = document.get("data") or {}
    for label, relative_file in data_mapping.items():
        dataset_path = root / relative_file
        dataset = xr.load_dataset(dataset_path)
        view = DatasetView(label=label, source_files={"dataset": str(dataset_path)})
        view.metadata.update({key: scalar_metadata(value) for key, value in dataset.attrs.items()})
        for raw_name, array in dataset.data_vars.items():
            raw_name = str(raw_name)
            semantic_name = SEMANTIC_VARIABLES.get(raw_name)
            canonical_array, transformations = canonicalize_array(array)
            if semantic_name is None:
                view.unmapped_fields.append(f"{dataset_path.name}:{raw_name}")
                view.raw_variables[raw_name] = canonical_array
                continue
            view.variables[semantic_name] = canonical_array
            view.transformations.extend(transformations)
        datasets[label] = view
    parameter_file = document.get("optimized_parameters")
    parameters = pd.read_csv(root / parameter_file) if parameter_file else None
    diagnostics = {
        key: scalar_metadata(value)
        for key, value in document.items()
        if key
        not in {
            "data",
            "optimized_parameters",
            "scheme",
            "initial_parameters",
            "parameter_history",
            "optimization_history",
        }
        and not isinstance(value, (dict, list))
    }
    scheme = _yaml(root / document["scheme"]) if document.get("scheme") else {}
    return ResultView(
        scenario=scenario or root.name,
        root=root,
        format="v0.7",
        datasets=datasets,
        parameters=parameters,
        diagnostics=diagnostics,
        provenance={"result_layout": "monolithic", "result_file": str(root / "result.yml")},
        scheme=scheme,
        unmapped_fields=[field for view in datasets.values() for field in view.unmapped_fields],
    )
