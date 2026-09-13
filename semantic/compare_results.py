"""Compare pinned pyglotaran results through an external semantic v0.7 view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import yaml
from compatibility.load_result import load_result
from compatibility.metrics import compare_arrays
from compatibility.normalize import scalar_metadata

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8
SEMANTIC_VARIABLES = ("data", "residual", "fitted_data", "clp", "matrix")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or Path(__file__).with_name("scenarios.yml")
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_run_manifest(result_root: Path) -> dict[str, Any] | None:
    """Load the runner manifest adjacent to a retained result tree, if present."""

    manifest_path = result_root.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def scenario_map(root: Path) -> dict[str, Path]:
    """Discover result leaves for compatibility with the earlier triage script."""

    scenarios: dict[str, Path] = {}
    for result_file in root.rglob("result.yml"):
        scenarios[result_file.parent.relative_to(root).as_posix()] = result_file.parent
    return scenarios


def dataset_labels(staging_scenario: Path) -> list[str]:
    document = yaml.safe_load((staging_scenario / "result.yml").read_text(encoding="utf-8")) or {}
    if "optimization_results" in document:
        return sorted((document.get("optimization_results") or {}).keys())
    return sorted((document.get("data") or {}).keys())


def locate_main_dataset(main_scenario: Path, label: str) -> Path | None:
    document = yaml.safe_load((main_scenario / "result.yml").read_text(encoding="utf-8")) or {}
    relative_file = (document.get("data") or {}).get(label)
    path = main_scenario / relative_file if relative_file else main_scenario / f"{label}.nc"
    return path if path.is_file() else None


def _parameter_metrics(
    expected: pd.DataFrame | None,
    current: pd.DataFrame | None,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if expected is None or current is None or "label" not in expected or "label" not in current:
        return {"status": "missing"}
    expected_series = expected.set_index("label")["value"]
    current_series = current.set_index("label")["value"]
    shared = expected_series.index.intersection(current_series.index)
    missing_in_main = sorted(set(current_series.index) - set(expected_series.index))
    missing_in_staging = sorted(set(expected_series.index) - set(current_series.index))
    if len(shared) == 0:
        return {
            "status": "different",
            "shared_count": 0,
            "missing_in_main": missing_in_main,
            "missing_in_staging": missing_in_staging,
        }
    expected_values = expected_series.loc[shared].to_numpy(dtype=float)
    current_values = current_series.loc[shared].to_numpy(dtype=float)
    difference = np.abs(expected_values - current_values)
    scale = np.maximum(np.abs(expected_values), np.finfo(float).eps)
    relative = difference / scale
    worst = int(np.argmax(relative))
    return {
        "status": "pass"
        if np.allclose(expected_values, current_values, rtol=rtol, atol=atol, equal_nan=True)
        and not missing_in_main
        and not missing_in_staging
        else "different",
        "shared_count": len(shared),
        "missing_in_main": missing_in_main,
        "missing_in_staging": missing_in_staging,
        "max_abs": float(np.max(difference)),
        "max_relative": float(np.max(relative)),
        "worst_label": str(shared[worst]),
        "worst_main_value": float(expected_values[worst]),
        "worst_staging_value": float(current_values[worst]),
    }


def _metadata_metrics(expected: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "dataset_scale": ("dataset_scale", "scale"),
        "root_mean_square_error": ("root_mean_square_error",),
        "weighted_root_mean_square_error": ("weighted_root_mean_square_error",),
    }
    for canonical, names in aliases.items():
        expected_value = next((expected[name] for name in names if name in expected), None)
        current_value = next((current[name] for name in names if name in current), None)
        if expected_value is None and current_value is None:
            continue
        if expected_value is None or current_value is None:
            result[canonical] = {
                "status": "missing",
                "main": expected_value,
                "staging": current_value,
            }
            continue
        difference = abs(float(expected_value) - float(current_value))
        result[canonical] = {
            "status": "pass"
            if np.isclose(expected_value, current_value, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
            else "different",
            "main": float(expected_value),
            "staging": float(current_value),
            "abs_difference": float(difference),
            "staging_source": current.get(f"{canonical}_source", "persisted"),
        }
    return result


def _scenario_result(
    main_root: Path, staging_root: Path, specification: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    scenario = str(specification["id"])
    relative_result = Path(specification["result"])
    main_path = main_root / relative_result
    staging_path = staging_root / relative_result
    record: dict[str, Any] = {
        "scenario": scenario,
        "notebook": specification["notebook"],
        "result": specification["result"],
        "expected_path": str(main_path),
        "staging_path": str(staging_path),
    }
    if not main_path.is_dir() or not (main_path / "result.yml").is_file():
        record.update(
            {"status": "BASELINE_FAILURE", "reason": "declared v0.7 result leaf is missing"}
        )
        return record
    if not staging_path.is_dir() or not (staging_path / "result.yml").is_file():
        record.update({"status": "REGRESSION", "reason": "declared v0.8 result leaf is missing"})
        return record
    for path, failure_status in ((main_path, "BASELINE_FAILURE"), (staging_path, "REGRESSION")):
        try:
            view = load_result(path, scenario)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            record.update(status=failure_status, reason=f"{path}: {error}")
            return record
        if failure_status == "BASELINE_FAILURE":
            expected = view
        else:
            current = view
    fit_tolerance = float(
        specification.get("fitted_data_normalized_rms", defaults["fitted_data_normalized_rms"])
    )
    parameter_rtol = float(specification.get("parameter_rtol", defaults["parameter_rtol"]))
    parameter_atol = float(specification.get("parameter_atol", defaults["parameter_atol"]))
    datasets: list[dict[str, Any]] = []
    failures: list[str] = []
    for label in sorted(set(expected.datasets) | set(current.datasets)):
        if label not in expected.datasets or label not in current.datasets:
            failures.append(f"dataset:{label}")
            datasets.append({"dataset": label, "status": "missing"})
            continue
        expected_dataset = expected.datasets[label]
        current_dataset = current.datasets[label]
        variables: dict[str, Any] = {}
        for variable in SEMANTIC_VARIABLES:
            if (
                variable not in expected_dataset.variables
                or variable not in current_dataset.variables
            ):
                variables[variable] = {"status": "missing"}
                if variable in expected_dataset.variables or variable in {"data", "fitted_data"}:
                    failures.append(f"{label}/{variable}")
                continue
            variable_rtol = 0.0 if variable == "data" else DEFAULT_RTOL
            variable_atol = 0.0 if variable == "data" else DEFAULT_ATOL
            variables[variable] = compare_arrays(
                expected_dataset.variables[variable],
                current_dataset.variables[variable],
                rtol=variable_rtol,
                atol=variable_atol,
            )
        if variables.get("data", {}).get("status") != "pass":
            failures.append(f"{label}/data")
        fit_metric = variables.get("fitted_data", {})
        if (
            fit_metric.get("status") in {"structural_mismatch", "nonfinite"}
            or not np.isfinite(fit_metric.get("normalized_rms", float("inf")))
            or fit_metric.get("normalized_rms", float("inf")) > fit_tolerance
        ):
            failures.append(f"{label}/fitted_data")
        datasets.append(
            {
                "dataset": label,
                "status": "compared",
                "variables": variables,
                "metadata": _metadata_metrics(expected_dataset.metadata, current_dataset.metadata),
                "transformations": sorted(
                    set(expected_dataset.transformations + current_dataset.transformations)
                ),
                "unmapped_fields": sorted(
                    set(expected_dataset.unmapped_fields + current_dataset.unmapped_fields)
                ),
            }
        )
    parameters = _parameter_metrics(
        expected.parameters,
        current.parameters,
        rtol=parameter_rtol,
        atol=parameter_atol,
    )
    requested_status = specification.get("difference_classification")
    if failures:
        status = "REGRESSION"
        reason = "; ".join(failures)
    elif requested_status:
        status = str(requested_status)
        reason = specification.get("difference_reason", "documented representation difference")
    elif parameters.get("status") != "pass":
        status = "EXPECTED_DIFFERENCE"
        reason = "parameter or decomposition differences remain after fitted-data agreement"
    else:
        status = "PASS"
        reason = "semantic fields meet the scenario acceptance contract"
    record.update(
        {
            "status": status,
            "reason": reason,
            "fitted_data_normalized_rms_tolerance": fit_tolerance,
            "parameter_tolerance": {"rtol": parameter_rtol, "atol": parameter_atol},
            "parameters": parameters,
            "datasets": datasets,
            "transformations": sorted(set(expected.transformations + current.transformations)),
            "unmapped_fields": sorted(set(expected.unmapped_fields + current.unmapped_fields)),
            "provenance": {"main": expected.provenance, "staging": current.provenance},
        }
    )
    return record


def compare_results(
    main_root: Path, staging_root: Path, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = load_manifest() if manifest is None else manifest
    specifications = manifest.get("scenarios", [])
    if not specifications or len({item["id"] for item in specifications}) != len(specifications):
        message = "Contract must declare a nonempty set of unique scenarios"
        raise ValueError(message)
    for item in specifications:
        tolerance = float(
            item.get(
                "fitted_data_normalized_rms",
                manifest.get("defaults", {}).get("fitted_data_normalized_rms", float("nan")),
            )
        )
        if not np.isfinite(tolerance) or tolerance < 0:
            message = "Fit tolerance must be finite and nonnegative"
            raise ValueError(message)
    defaults = manifest.get("defaults") or {}
    records = [
        _scenario_result(main_root, staging_root, item, defaults)
        for item in manifest.get("scenarios", [])
    ]
    statuses = ("PASS", "EXPECTED_DIFFERENCE", "REGRESSION", "BASELINE_FAILURE", "BLOCKED")
    counts = {status: sum(record["status"] == status for record in records) for status in statuses}
    return {
        "contract": manifest,
        "main_root": str(main_root),
        "staging_root": str(staging_root),
        "run_manifests": {
            "main": load_run_manifest(main_root),
            "staging": load_run_manifest(staging_root),
        },
        "scenarios": records,
        "summary": {
            "scenario_count": len(records),
            "status_counts": counts,
            "acceptable": counts["REGRESSION"] == 0
            and counts["BASELINE_FAILURE"] == 0
            and counts["BLOCKED"] == 0,
        },
    }


def _json_default(value: Any) -> Any:
    normalized = scalar_metadata(value)
    if normalized is not value:
        return normalized
    if isinstance(value, Path):
        return str(value)
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "## v0.7.4 vs v0.8 staging semantic comparison",
        "",
        (
            "Generated by validation/compare_results.py using the external "
            "validation/compatibility layer."
        ),
        "",
        "| Scenario | Status | Fit tolerance | Worst fitted-data normalized RMS | Parameters |",
        "|---|---|---:|---:|---|",
    ]
    for record in report["scenarios"]:
        fit_metrics = [
            dataset["variables"].get("fitted_data", {}).get("normalized_rms", 0.0)
            for dataset in record.get("datasets", [])
            if dataset.get("status") == "compared"
        ]
        worst_fit = max(fit_metrics, default=None)
        lines.append(
            f"| {record['scenario']} | **{record['status']}** | "
            f"{record.get('fitted_data_normalized_rms_tolerance', 'n/a')} | "
            f"{worst_fit if worst_fit is not None else 'n/a'} | "
            f"{record.get('parameters', {}).get('status', 'n/a')} |"
        )
    lines += [
        "",
        f"Acceptable comparison: **{report['summary']['acceptable']}**.",
        "",
        (
            "PASS means the semantic acceptance contract is met. "
            "EXPECTED_DIFFERENCE is reserved for a documented, tested "
            "representation or identifiability difference after fitted-data agreement. "
            "Missing declared artifacts are hard failures."
        ),
        "",
        "### Remaining documented differences",
        "",
    ]
    for record in report["scenarios"]:
        if record["status"] == "EXPECTED_DIFFERENCE":
            lines.append(f"- {record['scenario']}: {record['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_results(
        args.main_root.resolve(), args.staging_root.resolve(), load_manifest(args.manifest)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    markdown_report = markdown(report)
    args.output.with_suffix(".md").write_text(markdown_report, encoding="utf-8")
    sys.stdout.write(markdown_report)
    return 0 if report["summary"]["acceptable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
