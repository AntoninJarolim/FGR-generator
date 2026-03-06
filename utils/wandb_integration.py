"""
Wandb integration for eval: register stats and log one comparison Table per run.
No file I/O or hashing here; caller decides when data is new and calls register_for_log.
Runs are keyed by name: same name resumes the same run so new plots are added instead of creating duplicates.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

try:
    import wandb
except ImportError:
    print("Warning: Wandb integration requires wandb to be installed.")
    wandb = None


@dataclass
class PendingEntry:
    """One method's registered stats for a given input_dir, used to build a comparison Table."""

    method_name: str
    statistics: dict


_pending: dict[str, list[PendingEntry]] = {}


def register_for_log(
    input_dir: str,
    method_name: str,
    statistics: dict,
) -> None:
    """Register this method's stats for wandb logging. Call only when data is new (caller does hash check + save)."""
    normalized = input_dir.removeprefix("data/").lstrip("/") or "eval"
    _pending.setdefault(normalized, []).append(
        PendingEntry(method_name=method_name, statistics=statistics)
    )


def _build_comparison_table(pending_list: list[PendingEntry]) -> Any:
    """Build a wandb Table: one row per method. Standard is first; each entry has one method."""
    if wandb is None:
        return None

    # Caller registers standard first, then others. Each entry has exactly one method in statistics.
    by_method: dict[str, dict] = {}
    n_valid: int | None = None
    for entry in pending_list:
        avgs = entry.statistics.get("averages") or {}
        counts = entry.statistics.get("counts") or {}
        method = entry.method_name
        if method in avgs:
            by_method[method] = {
                "avg_f1": avgs[method]["avg_f1"],
                "avg_em": avgs[method]["avg_em"],
            }
        if entry.method_name == "standard" and "valid" in counts:
            n_valid = counts["valid"]
        elif entry.method_name != "standard":
            by_method.setdefault(method, {})["n_diff_pred"] = counts.get("different_prediction")

    methods_ordered = ["standard"] + sorted(by_method.keys() - {"standard"})

    def _cell(v: Any, fmt: str = "str") -> str:
        if v is None:
            return "-"
        return f"{v:.4f}" if fmt == "f4" else str(v)

    rows = []
    for method in methods_ordered:
        m = by_method.get(method, {})
        n_diff_pred = "-" if method == "standard" else m.get("n_diff_pred")
        rows.append([
            method,
            _cell(m.get("avg_f1"), "f4"),
            _cell(m.get("avg_em"), "f4"),
            _cell(n_valid),
            _cell(n_diff_pred),
        ])

    return wandb.Table(
        columns=[
            "method",
            "avg_f1",
            "avg_em",
            "n_valid",
            "n_different_prediction_vs_standard",
        ],
        data=rows,
    )


def _run_id_from_name(name: str) -> str:
    """Stable run id from run name so the same name always maps to the same run (for resume)."""
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def log_run(input_dir: str, project: str = "fgr-eval") -> None:
    """
    If any method was registered for this input_dir: init wandb and log one comparison Table.
    Call once at the end of evaluation.
    Uses run name as identifier: same input_dir resumes the same run so new tables/plots are added
    instead of creating multiple runs with the same name.
    """
    normalized = input_dir.removeprefix("data/").lstrip("/") or "eval"
    pending_list = _pending.pop(normalized, None)
    if not pending_list:
        return
    if wandb is None:
        return

    run_id = _run_id_from_name(normalized)
    wandb.init(
        project=project,
        id=run_id,
        name=normalized,
        resume="allow",
        config={"input_dir": input_dir},
    )
    table = _build_comparison_table(pending_list)
    if table is not None:
        wandb.log({"eval/comparison_table": table})
