"""Loader for the pinned v0.8 split result layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from .metrics import weighted_root_mean_square_error
from .normalize import canonicalize_array
from .normalize import first_data_array
from .normalize import scalar_metadata
from .schema import DatasetView
from .schema import ResultView
from .weights import reconstruct_weight


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_field(path: Path, preferred: str | None = None) -> tuple[xr.DataArray, list[str]]:
    dataset = xr.load_dataset(path)
    array = first_data_array(dataset, preferred)
    array, transformations = canonicalize_array(array)
    return array, transformations


def load_v08_result(root: str | Path, scenario: str | None = None) -> ResultView:
    root = Path(root)
    document = _yaml(root / "result.yml")
    optimization_results = document.get("optimization_results") or {}
    datasets: dict[str, DatasetView] = {}
    for label, entry in optimization_results.items():
        dataset_root = root / "optimization_results" / label
        view = DatasetView(label=label)
        for semantic_name, entry_name, preferred in (
            ("data", "input_data", "data"),
            ("residual", "residuals", "residual"),
            ("fitted_data", "fitted_data", None),
            ("clp", "fit_decomposition/clp", None),
            ("matrix", "fit_decomposition/matrix", None),
        ):
            subdirectory = ""
            relative_file = entry.get(entry_name) if "/" not in entry_name else None
            if "/" in entry_name:
                group, name = entry_name.split("/", 1)
                relative_file = (entry.get(group) or {}).get(name)
                subdirectory = group
            if not relative_file:
                view.unmapped_fields.append(f"missing:{entry_name}")
                continue
            path = (
                dataset_root
                / subdirectory
                / (relative_file if isinstance(relative_file, str) else str(relative_file))
            )
            if not path.is_file():
                view.unmapped_fields.append(f"missing:{path}")
                continue
            array, transformations = _load_field(path, preferred)
            view.variables[semantic_name] = array
            view.source_files[semantic_name] = str(path)
            view.transformations.extend(transformations)
            for key, value in array.attrs.items():
                view.metadata.setdefault(key, scalar_metadata(value))
        meta = entry.get("meta") or {}
        view.metadata.update({key: scalar_metadata(value) for key, value in meta.items()})
        for semantic_name, entry_name in (("element", "elements"), ("activation", "activations")):
            for name, relative_file in (entry.get(entry_name) or {}).items():
                if not relative_file:
                    view.unmapped_fields.append(f"missing:{entry_name}/{name}")
                    continue
                path = dataset_root / entry_name / relative_file
                if not path.is_file():
                    view.unmapped_fields.append(f"missing:{path}")
                    continue
                dataset = xr.load_dataset(path)
                for raw_name, array in dataset.data_vars.items():
                    canonical_array, transformations = canonicalize_array(array)
                    view.variables[f"{semantic_name}:{name}:{raw_name}"] = canonical_array
                    view.raw_variables[f"{semantic_name}:{name}:{raw_name}"] = canonical_array
                    view.transformations.extend(transformations)
        residual = view.variables.get("residual")
        if (
            residual is not None
            and "dataset_scale" not in view.metadata
            and "scale" not in view.metadata
        ):
            view.metadata["dataset_scale"] = 1.0
            view.metadata["dataset_scale_source"] = "derived_default_scale"
            view.transformations.append("metadata:dataset_scale=default:1")
        if residual is not None and "weighted_root_mean_square_error" not in view.metadata:
            weight = view.variables.get("weight")
            if weight is None:
                weight = reconstruct_weight(
                    residual,
                    _yaml(root / document["scheme"]) if document.get("scheme") else {},
                    label,
                )
            if weight is None:
                weight = xr.ones_like(residual, dtype=float)
                view.metadata["weighted_root_mean_square_error_source"] = (
                    "derived_from_default_weight"
                )
            derived = weighted_root_mean_square_error(residual, weight)
            if derived is not None:
                view.metadata["weighted_root_mean_square_error"] = derived
                view.metadata.setdefault(
                    "weighted_root_mean_square_error_source", "derived_from_result_weight"
                )
                view.transformations.append("metadata:weighted_root_mean_square_error=derived")
        datasets[label] = view
    parameter_file = document.get("optimized_parameters")
    parameters = pd.read_csv(root / parameter_file) if parameter_file else None
    optimization_info = document.get("optimization_info") or {}
    diagnostics = {
        key: scalar_metadata(value)
        for key, value in optimization_info.items()
        if not isinstance(value, (dict, list))
    }
    scheme = _yaml(root / document["scheme"]) if document.get("scheme") else {}
    return ResultView(
        scenario=scenario or root.name,
        root=root,
        format="v0.8",
        datasets=datasets,
        parameters=parameters,
        diagnostics=diagnostics,
        provenance={"result_layout": "split", "result_file": str(root / "result.yml")},
        scheme=scheme,
        unmapped_fields=[field for view in datasets.values() for field in view.unmapped_fields],
    )
