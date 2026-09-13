"""Reconstruct explicit model weights when a result file omitted them."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterable


def _weight_rules(scheme: dict[str, Any], dataset_label: str) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in scheme.get("weights", []) or []:
        datasets = rule.get("datasets")
        if datasets is None or dataset_label in datasets:
            rules.append(rule)
    for experiment in (scheme.get("experiments") or {}).values():
        dataset = (experiment.get("datasets") or {}).get(dataset_label, {})
        rules.extend(dataset.get("weights", []) or [])
    return rules


def _interval_slice(axis: np.ndarray, interval: Iterable[float]) -> slice:
    bounds = list(interval)
    if len(bounds) != 2:
        message = f"Weight interval must have two bounds, got {bounds!r}"
        raise ValueError(message)
    low, high = sorted(bounds)
    start = 0 if np.isneginf(low) else int(np.abs(axis - low).argmin())
    stop = axis.size if np.isposinf(high) else int(np.abs(axis - high).argmin()) + 1
    return slice(start, stop)


def reconstruct_weight(
    residual: xr.DataArray, scheme: dict[str, Any], dataset_label: str
) -> xr.DataArray | None:
    """Build a weight array from v0.7 or v0.8 scheme YAML, if explicit rules exist."""

    rules = _weight_rules(scheme, dataset_label)
    if not rules:
        return None
    weight = xr.ones_like(residual, dtype=float)
    model_dimension = str(residual.attrs.get("model_dimension", residual.dims[0]))
    global_dimension = str(residual.attrs.get("global_dimension", residual.dims[-1]))
    for rule in rules:
        indexers: dict[Hashable, slice] = {}
        if rule.get("global_interval") is not None and global_dimension in residual.dims:
            indexers[global_dimension] = _interval_slice(
                np.asarray(residual.coords[global_dimension]), rule["global_interval"]
            )
        if rule.get("model_interval") is not None and model_dimension in residual.dims:
            indexers[model_dimension] = _interval_slice(
                np.asarray(residual.coords[model_dimension]), rule["model_interval"]
            )
        slices = tuple(indexers.get(dimension, slice(None)) for dimension in weight.dims)
        weight.data[slices] *= float(rule["value"])
    return weight
