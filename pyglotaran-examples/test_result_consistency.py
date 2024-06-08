"""Tests to ensure result consistency."""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Protocol
from warnings import warn

import numpy as np
import pandas as pd
import pytest
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from xarray.core.coordinates import DataArrayCoordinates


HERE = Path(__file__).parent
RUN_EXAMPLES_MSG = (
    "run 'python scripts/run_examples.py run-all --headless' "
    "in the 'pyglotaran-examples' repo root."
)

# in general this list should be empty, but for none stable examples this might be needed
EXAMPLE_BLOCKLIST = [
    "study_transient_absorption_two_dataset_analysis_result_2d_co_co2",
    "ex_spectral_guidance",
]
ALLOW_MISSING_COORDS = {"spectral": ("matrix", "species_concentration")}

SVD_PATTERN = re.compile(r"(?P<pre_fix>.+?)(right|left)_singular_vectors")

MISSING_RESULT_FILES = set()


class AllCloseFixture(Protocol):
    def __call__(
        self,
        a: float | np.ndarray | xr.DataArray,
        b: float | np.ndarray | xr.DataArray,
        rtol: float | np.ndarray = 1e-5,
        atol: float | np.ndarray = 1e-8,
        xtol: int = 0,
        equal_nan: bool = False,
        print_fail: int = 5,
        record_rmse: bool = True,
    ) -> bool: ...


class GitError(Exception):
    """Error raised when a git interaction didn't exit with a 0 returncode."""


def get_compare_results_path() -> Path:
    """Ensure that the comparison-results exist, are up to date and return their path."""
    compare_result_folder = HERE / "comparison-results"
    example_repo = "https://github.com/glotaran/pyglotaran-examples.git"
    if not compare_result_folder.exists():
        proc_clone = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "-b",
                "comparison-results",
                example_repo,
                str(compare_result_folder),
            ],
            capture_output=True,
        )
        if proc_clone.returncode != 0:
            msg = f"Error cloning {example_repo}:\n{proc_clone.stderr.decode()}"
            raise GitError(msg)
    if "GITHUB" not in os.environ:
        proc_fetch = subprocess.run(
            [
                "git",
                "-C",
                compare_result_folder.as_posix(),
                "fetch",
                "--depth",
                "1",
                "origin",
                "comparison-results",
            ],
            capture_output=True,
        )
        if proc_fetch.returncode != 0:
            msg = f"Error fetching {example_repo}:\n{proc_fetch.stderr.decode()}"
            raise GitError(msg)
        proc_reset = subprocess.run(
            [
                "git",
                "-C",
                compare_result_folder.as_posix(),
                "reset",
                "--hard",
                "origin/comparison-results",
            ],
            capture_output=True,
        )
        if proc_reset.returncode != 0:
            msg = f"Error resetting {example_repo}:\n{proc_reset.stderr.decode()}"
            raise GitError(msg)
    return compare_result_folder


def get_current_result_path() -> Path:
    """Get the path of the current results."""
    local_path = Path.home() / "pyglotaran_examples_results"
    ci_path = Path(os.getenv("GITHUB_WORKSPACE", "")) / "comparison-results-current"
    if local_path.exists():
        return local_path
    if ci_path.exists():
        return ci_path
    msg = f"No current results present at {local_path} or {ci_path}, {RUN_EXAMPLES_MSG}"
    raise ValueError(msg)


def rename_with_suffix(
    expected_name: str, suffixed_names: Iterable[str], current_keys: Iterable[str]
) -> str:
    """Replace ``expected_name`` with the suffixed version in the dataset keys.

    Parameters
    ----------
    expected_name: str
        Expected name of a variable (data_var, coord)
    suffixed_names: list[str]
        Names that are allowed/expected to have suffixes.
    current_keys: Iterable[str]
        Keys of the current dataset.

    Returns
    -------
    str
        Updated expected var name.
    """
    if expected_name in suffixed_names:
        return next((key for key in current_keys if key.startswith(expected_name)), expected_name)
    return expected_name


def coord_test(
    expected_coords: DataArrayCoordinates,
    current_coords: DataArrayCoordinates,
    file_name: str,
    allclose: AllCloseFixture,
    exact_match=False,
    data_var_name: str = "unknown",
) -> None:
    """Run tests that coordinates are exactly equal if string coords or close."""
    __tracebackhide__ = True
    for expected_coord_name, expected_coord_value in expected_coords.items():
        if (
            expected_coord_name in ALLOW_MISSING_COORDS
            and data_var_name in ALLOW_MISSING_COORDS[expected_coord_name]
        ):
            print(f"- allow missing coordinate: {expected_coord_name} in variable {data_var_name}")
            continue

        expected_coord_name = rename_with_suffix(
            expected_coord_name,
            [
                "species",
                "initial_concentration",
                "component",
                "rate",
                "lifetime",
                "to_species",
                "from_species",
            ],
            current_coords.keys(),
        )

        current_coord_value = current_coords[expected_coord_name]

        assert expected_coord_name in current_coords, (
            f"Missing coordinate: {expected_coord_name!r} in {file_name!r}, "
            f"data_var {data_var_name!r}"
        )

        if (
            exact_match
            or np.issubdtype(expected_coord_value.data.dtype, np.str_)
            or np.issubdtype(expected_coord_value.data.dtype, object)
        ):
            assert np.array_equal(expected_coord_value, current_coord_value), (
                f"Coordinate value mismatch in {file_name!r}, "
                f"data_var {data_var_name!r} and {expected_coord_name=}"
            )
        else:
            assert allclose(expected_coord_value, current_coord_value, rtol=1e-5, print_fail=20), (
                f"Coordinate value mismatch in {file_name!r}, "
                f"data_var {data_var_name!r} and {expected_coord_name=}"
            )


def calculate_singular_vectors_compare_sign(
    expected_result: xr.Dataset, current_result: xr.Dataset, data_var_name: str
) -> xr.DataArray:
    """Calculate the signs of expected vs. current singular vectors for a given ``data_var_name``.

    Note:
    -----
    We sum long the non SV index dimension to prevent false positive detecting of sign flips in
    the numerical noise.

    Parameters
    ----------
    expected_result : xr.Dataset
        Expected result dataset.
    current_result : xr.Dataset
        Current result dataset.
    data_var_name : str
        Name of the singular vector data variable.

    Returns
    -------
    xr.DataArray
        1D DataArray with dimension of the singular vectors index.
    """
    assert "singular_vectors" in data_var_name
    expected_values = expected_result.data_vars[data_var_name]
    sum_dim = next(
        dim_name for dim_name in expected_values.dims if "singular_value" not in dim_name
    )
    return np.sign(expected_values.sum(dim=sum_dim)) * np.sign(
        current_result.data_vars[data_var_name].sum(dim=sum_dim)
    )


def singular_vectors_compare_signs(
    expected_result: xr.Dataset, current_result: xr.Dataset, expected_var_name: str, pre_fix: str
) -> xr.DataArray:
    """Calculate the signs of expected vs. current singular vectors for ``expected_var_name``.

    This function is needed due to the fact the sign of singular vectors in singular value
    decomposition can flip because the choice of sign in the singular vectors is arbitrary and can
    be multiplied by -1 without affecting the validity of the decomposition.

    In addition to ``calculate_singular_vectors_compare_sign`` this function also performs a
    sanity check that the signs of LSV and the corresponding RSV comparing the current vs. expected
    results compensate each other.

    Parameters
    ----------
    expected_result : xr.Dataset
        Expected result dataset.
    current_result : xr.Dataset
        Current result dataset.
    expected_var_name : str
        Name of the singular vector data variable.
    pre_fix : str
        Prefix of ``expected_var_name``.

    Returns
    -------
    xr.DataArray
        1D DataArray with dimension of the singular vectors index.

    See Also
    --------
    calculate_singular_vectors_compare_sign
    """
    __tracebackhide__ = True
    lsv_signs = calculate_singular_vectors_compare_sign(
        expected_result, current_result, f"{pre_fix}left_singular_vectors"
    )
    rsv_signs = calculate_singular_vectors_compare_sign(
        expected_result, current_result, f"{pre_fix}right_singular_vectors"
    )
    # Sanity check
    # Since we don't save the full matrix we can assume lsv_signs.size == rsv_signs.size and
    # thus check that that the flipped signs are at the same position.
    assert all(lsv_signs.to_numpy() == rsv_signs.to_numpy())
    return lsv_signs if "left" in expected_var_name else rsv_signs


def data_var_test(
    allclose: AllCloseFixture,
    expected_result: xr.Dataset,
    current_result: xr.Dataset,
    file_name: str,
    expected_var_name: str,
) -> None:
    """Run test that a data_var of the current_result is close to the expected_result."""
    __tracebackhide__ = True
    expected_values = expected_result.data_vars[expected_var_name]

    # weighted_data were always calculated and now will only be calculated
    # when weights are applied
    if expected_var_name == "weighted_data" and expected_var_name not in current_result.data_vars:
        return

    # weight related data vars are only saved if the data has weights applied
    if "weight" in expected_var_name and "weight" not in expected_result.data_vars:
        return

    expected_var_name = rename_with_suffix(
        expected_var_name,
        [
            "decay_associated_spectra",
            "decay_associated_images",
            "a_matrix",
            "k_matrix",
            "k_matrix_reduced",
        ],
        current_result.data_vars.keys(),
    )

    assert (
        expected_var_name in current_result.data_vars
    ), f"Missing data_var: {expected_var_name!r} in {file_name!r}"
    current_values = current_result.data_vars[expected_var_name]

    eps = np.finfo(np.float32).eps
    rtol = 1e-5  # default value of allclose
    if expected_var_name.endswith("residual"):  # type:ignore[operator]
        eps = max(eps, expected_result["data"].to_numpy().max() * eps)

    if "singular_vectors" in expected_var_name:  # type:ignore[operator]
        # Sometimes the coords in the (right) singular vectors are swapped
        if expected_values.dims != current_values.dims:
            warn(
                dedent(
                    f"""\n
                    Dimensions transposed for {expected_var_name!r} in {file_name!r}.
                    - expected: {expected_values.dims}
                    - current:  {current_values.dims}
                    """
                ),
                stacklevel=2,
            )
            expected_values = expected_values.transpose(*current_values.dims)
        rtol = 1e-4  # instead of 1e-5
        eps = 1e-5  # type:ignore[assignment] # instead of ~1.2e-7
        pre_fix = SVD_PATTERN.match(expected_var_name).group(  # type:ignore[union-attr]
            "pre_fix"
        )
        expected_singular_values = expected_result.data_vars[f"{pre_fix}singular_values"]
        expected_values = expected_values * singular_vectors_compare_signs(
            expected_result, current_result, expected_var_name, pre_fix
        )

        if expected_values.shape[0] == expected_singular_values.shape[0]:
            expected_values_scaled = np.diag(expected_singular_values).dot(expected_values.data)
        else:
            expected_values_scaled = expected_values.data.dot(np.diag(expected_singular_values))

        float_resolution = np.maximum(
            np.abs(eps * expected_values_scaled),
            np.ones(expected_values.data.shape) * eps,
        )
    elif "spectra" in expected_var_name:
        float_resolution = np.maximum(
            np.ones(expected_values.data.shape) * eps * np.max(np.abs(expected_values.data)),
            np.ones(expected_values.data.shape) * eps,
        )
    else:
        float_resolution = np.maximum(
            np.abs(eps * expected_values.data),
            np.ones(expected_values.data.shape) * eps,
        )

    abs_diff = np.abs(expected_values - current_values)

    assert allclose(
        expected_values,
        current_values,
        atol=float_resolution,
        rtol=rtol,
        print_fail=20,
    ), (
        f"Result data_var data mismatch: {expected_var_name!r} in {file_name!r}.\n"
        "With sum of absolute difference: "
        f"{float(np.sum(abs_diff))} and shape: {expected_values.shape}\n"
        "Mean difference: "
        f"{float(np.sum(abs_diff))/np.prod(expected_values.shape)}\n"
        f"Using: \n - {rtol=} \n - {eps=} \n - {float_resolution=}"
    )

    coord_test(
        expected_values.coords,
        current_values.coords,
        file_name,
        allclose,
        data_var_name=expected_var_name,  # type:ignore[operator]
    )


def map_result_files(file_glob_pattern: str) -> dict[str, list[tuple[Path, Path]]]:
    """Load all datasets and map them in a dict."""
    result_map = defaultdict(list)
    if (compare_results_path_str := os.getenv("COMPARE_RESULTS_LOCAL")) is not None:
        compare_results_path = Path(compare_results_path_str)
        warn(
            dedent(
                f"""
                Using Path in environment variable COMPARE_RESULTS_LOCAL:
                {compare_results_path.as_posix()}
                """
            ),
            stacklevel=2,
        )
        try:
            if not compare_results_path.exists():
                raise FileNotFoundError(
                    dedent(
                        f"""
                        Path in COMPARE_RESULTS_LOCAL not valid:
                        {compare_results_path}  <- does not exist
                        """
                    )
                )
        except OSError as exception:
            if str(compare_results_path).startswith(('"', "'")):
                msg = "Path in COMPARE_RESULTS_LOCAL should not start with ' or \""
                raise ValueError(msg) from exception
            raise exception
    else:
        compare_results_path = get_compare_results_path()
    current_result_path = get_current_result_path()
    for expected_result_file in compare_results_path.rglob(file_glob_pattern):
        if expected_result_file.name == "parameter_history.csv":
            continue
        key = (
            expected_result_file.relative_to(compare_results_path)
            .parent.as_posix()
            .replace("/", "_")
        )
        if key in EXAMPLE_BLOCKLIST:
            continue
        current_result_file = current_result_path / expected_result_file.relative_to(
            compare_results_path
        )
        if current_result_file.is_file() is False and "parameters" in current_result_file.name:
            current_result_file = (
                current_result_file.parent
                / f"parameters_{current_result_file.name.replace('_parameters','')}"
            )
        if current_result_file.exists():
            result_map[key].append((expected_result_file, current_result_file))
        else:
            MISSING_RESULT_FILES.add(
                expected_result_file.relative_to(compare_results_path).as_posix()
            )
            warn(
                UserWarning(
                    f"No current result for: {expected_result_file.as_posix()}, {RUN_EXAMPLES_MSG}"
                ),
                stacklevel=2,
            )
    return result_map


@lru_cache(maxsize=1)
def map_result_data() -> tuple[dict[str, list[tuple[xr.Dataset, xr.Dataset, str]]], set[str]]:
    """Load all datasets and map them in a tuple of dict and set of data_var names."""
    result_map = defaultdict(list)
    data_var_names = set()
    result_file_map = map_result_files(file_glob_pattern="*.nc")
    for key, path_list in result_file_map.items():
        for expected_result_file, current_result_file in path_list:
            expected_result: xr.Dataset = xr.open_dataset(expected_result_file)
            result_map[key].append(
                (
                    expected_result,
                    xr.open_dataset(current_result_file),
                    expected_result_file.name,
                )
            )
            for data_var_name in expected_result.data_vars:
                if data_var_name != "data":
                    data_var_names.add(data_var_name)
    return result_map, data_var_names


@lru_cache(maxsize=1)
def map_result_parameters() -> dict[str, list[pd.DataFrame]]:
    """Load all optimized parameter files and map them in a dict."""

    result_map = defaultdict(list)
    result_file_map = map_result_files(file_glob_pattern="*parameters.csv")
    for key, path_list in result_file_map.items():
        for expected_result_file, current_result_file in path_list:
            compare_df = pd.DataFrame(
                {
                    "expected": pd.read_csv(expected_result_file, index_col="label")["value"],
                    "current": pd.read_csv(current_result_file, index_col="label")["value"],
                }
            )
            result_map[key].append(compare_df)
    return result_map


@pytest.mark.parametrize("result_name", map_result_data()[0].keys())
def test_original_data_exact_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """The original data need to be exactly the same."""
    for expected_result, current_result, file_name in map_result_data()[0][result_name]:
        assert np.array_equal(
            expected_result.data.data, current_result.data.data
        ), f"Original data mismatch: {result_name!r} in {file_name!r}"
        coord_test(
            expected_result.data.coords,
            current_result.data.coords,
            file_name,
            allclose,
            exact_match=True,
            data_var_name="data",
        )


@pytest.mark.parametrize("result_name", map_result_parameters().keys())
def test_result_parameter_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """Optimized parameters need to be approximately the same"""
    for compare_df in map_result_parameters()[result_name]:
        assert allclose(
            compare_df["expected"].values, compare_df["current"].values, print_fail=20
        ), f"Parameter Mismatch: {compare_df.index}"


@pytest.mark.parametrize("result_name", map_result_data()[0].keys())
def test_result_attr_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """Result dataset attributes need to be approximately the same."""
    for expected, current, file_name in map_result_data()[0][result_name]:
        for expected_attr_name, expected_attr_value in expected.attrs.items():
            if expected_attr_name == "source_path":
                continue
            assert (
                expected_attr_name in current.attrs
            ), f"Missing result attribute: {expected_attr_name!r} in {file_name!r}"

            if isinstance(expected_attr_value, str):
                assert expected_attr_value == current.attrs[expected_attr_name], expected_attr_name
            else:
                assert allclose(
                    expected_attr_value, current.attrs[expected_attr_name], print_fail=20
                ), f"Result attr value mismatch: {expected_attr_name!r} in {file_name!r}"


@pytest.mark.parametrize("expected_var_name", map_result_data()[1])
@pytest.mark.parametrize("result_name", map_result_data()[0].keys())
def test_result_data_var_consistency(
    allclose: AllCloseFixture, result_name: str, expected_var_name: str
):
    """Result dataset data variables need to be approximately the same."""
    for expected_result, current_result, file_name in map_result_data()[0][result_name]:
        if expected_var_name in expected_result.data_vars:
            data_var_test(allclose, expected_result, current_result, file_name, expected_var_name)


def test_all_result_files_found():
    """Check that there were no missing files."""
    error_str = "\n".join(MISSING_RESULT_FILES)
    assert (
        len(MISSING_RESULT_FILES) == 0
    ), f"Missing files in {get_current_result_path().as_posix()}: \n{error_str}"


if __name__ == "__main__":
    pytest.main([__file__])
