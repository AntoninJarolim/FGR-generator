"""
Wandb integration for eval: register stats and log one comparison Table per run.
No file I/O or hashing here; caller decides when data is new and calls register_for_log.
Always appends to the same run by name: we look up via the API if a run with that name
exists; if yes we resume by id and log, otherwise we create a new run (wandb assigns id).
"""

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


# Columns for the comparison hierarchy: we log only "different_*" (same = parent - different).
_DIFF_COUNT_KEYS = [
    "different_tokenization",
    "different_raw_output",
    "different_prediction",
    "different_ctx_tokens_before_start",
]


def _build_comparison_table(pending_list: list[PendingEntry]) -> Any:
    """Build a wandb Table: one row per method with metrics and full diff hierarchy (only different_* logged)."""
    if wandb is None:
        return None

    by_method: dict[str, dict] = {}
    for entry in pending_list:
        avgs = entry.statistics.get("averages") or {}
        counts = entry.statistics.get("counts") or {}
        method = entry.method_name
        if method in avgs:
            by_method[method] = {
                "avg_f1": avgs[method]["avg_f1"],
                "avg_em": avgs[method]["avg_em"],
                "all": counts.get("all"),
                "invalid": counts.get("invalid"),
                "valid": counts.get("valid"),
            }
            for key in _DIFF_COUNT_KEYS:
                by_method[method][key] = counts.get(key)

    methods_ordered = ["standard"] + sorted(by_method.keys() - {"standard"})

    def _cell(v: Any, fmt: str = "str") -> str:
        if v is None:
            return "-"
        return f"{v:.4f}" if fmt == "f4" else str(v)

    columns = [
        "method",
        "avg_f1",
        "avg_em",
        "all",
        "invalid",
        "valid",
        "diff_tokenization",
        "diff_raw_output",
        "diff_prediction",
        "diff_ctx_tokens_before_start",
    ]
    rows = []
    for method in methods_ordered:
        m = by_method.get(method, {})
        row = [
            method,
            _cell(m.get("avg_f1"), "f4"),
            _cell(m.get("avg_em"), "f4"),
            _cell(m.get("all")),
            _cell(m.get("invalid")),
            _cell(m.get("valid")),
        ]
        for key in _DIFF_COUNT_KEYS:
            row.append(_cell(m.get(key)))
        rows.append(row)

    return wandb.Table(columns=columns, data=rows)


def _get_run_id_by_name(project: str, name: str) -> str | None:
    """Return the id of the first run in the project with the given name, or None if not found."""
    if wandb is None:
        return None
    try:
        api = getattr(wandb, "Api", lambda: None)()
        if api is None:
            return None
        for run in api.runs(project, per_page=500):
            if getattr(run, "name", None) == name:
                return getattr(run, "id", None)
        return None
    except Exception:
        return None


def log_run(input_dir: str, project: str = "fgr-eval") -> None:
    """
    If any method was registered for this input_dir: init wandb and log one comparison Table.
    Call once at the end of evaluation.
    Looks up a run by name; if found we resume by its id and append, else we create a new run
    (no id passed so wandb assigns its own).
    """
    normalized = input_dir.removeprefix("data/").lstrip("/") or "eval"
    pending_list = _pending.pop(normalized, None)
    if not pending_list:
        return
    if wandb is None:
        return

    existing_id = _get_run_id_by_name(project, normalized)

    if existing_id is not None:
        wandb.init(
            project=project,
            id=existing_id,
            name=normalized,
            resume="allow",
            config={"input_dir": input_dir},
        )
    else:
        wandb.init(
            project=project,
            name=normalized,
            config={"input_dir": input_dir},
        )
    table = _build_comparison_table(pending_list)
    if table is not None:
        wandb.log({"eval/comparison_table": table})

    wandb.finish()
