"""Canonical naming and coordinate alignment for cross-version results."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

DIMENSION_ALIASES = {"amplitude_label": "clp_label"}
VARIABLE_ALIASES = {"amplitude": "clp"}


def canonicalize_array(array: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Return an array with v0.7-compatible dimension and variable aliases."""

    transformations: list[str] = []
    rename_dims = {
        old: new
        for old, new in DIMENSION_ALIASES.items()
        if old in array.dims and new not in array.dims
    }
    if rename_dims:
        array = array.rename(rename_dims)
        transformations.extend(f"dimension:{old}->{new}" for old, new in rename_dims.items())
    if array.name in VARIABLE_ALIASES:
        old_name = array.name
        array = array.rename(VARIABLE_ALIASES[old_name])
        transformations.append(f"variable:{old_name}->{array.name}")
    return array, transformations


def align_arrays(
    expected: xr.DataArray, current: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Align by named dimensions and coordinate labels, never by position alone."""

    expected, expected_transforms = canonicalize_array(expected)
    current, current_transforms = canonicalize_array(current)
    transformations = expected_transforms + current_transforms
    if set(expected.dims) != set(current.dims):
        return expected, current, transformations
    if expected.dims != current.dims:
        current = current.transpose(*expected.dims)
        transformations.append("dimension-order:current->expected")
    if expected.shape != current.shape:
        return expected, current, transformations
    for dimension in expected.dims:
        if dimension not in expected.coords or dimension not in current.coords:
            continue
        expected_coord = np.asarray(expected.coords[dimension].values)
        current_coord = np.asarray(current.coords[dimension].values)
        if expected_coord.shape != current_coord.shape or not np.array_equal(
            expected_coord, current_coord
        ):
            try:
                current = current.sel({dimension: expected.coords[dimension]})
                transformations.append(f"coordinate-order:{dimension}->expected")
            except (KeyError, ValueError):
                return expected, current, transformations
    return expected, current, transformations


def first_data_array(dataset: xr.Dataset, preferred: str | None = None) -> xr.DataArray:
    if preferred is not None and preferred in dataset.data_vars:
        return dataset[preferred]
    if not dataset.data_vars:
        message = "NetCDF file contains no data variables"
        raise ValueError(message)
    return dataset[next(iter(dataset.data_vars))]


def scalar_metadata(value: Any) -> Any:
    """Convert numpy scalar values to JSON-friendly Python values."""

    return value.item() if hasattr(value, "item") else value
