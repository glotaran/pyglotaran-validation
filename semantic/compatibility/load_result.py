"""Select the external adapter from the persisted result schema, not installed packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from .load_v07 import load_v07_result
from .load_v08 import load_v08_result

if TYPE_CHECKING:
    from pathlib import Path


def load_result(root: Path, scenario: str | None = None):
    document = yaml.safe_load((root / "result.yml").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "Invalid result document"
        raise ValueError(message)
    layouts = [key for key in ("data", "optimization_results") if key in document]
    if len(layouts) != 1 or not document[layouts[0]]:
        message = "Unknown, ambiguous, or empty result layout"
        raise ValueError(message)
    if not document.get("optimized_parameters"):
        message = "Missing optimized_parameters artifact declaration"
        raise ValueError(message)
    loader = load_v07_result if layouts[0] == "data" else load_v08_result
    view = loader(root, scenario)
    missing = [field for field in view.unmapped_fields if field.startswith("missing:")]
    if missing:
        raise ValueError("Missing artifacts: " + ", ".join(missing))
    return view
