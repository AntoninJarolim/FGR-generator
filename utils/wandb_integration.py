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
