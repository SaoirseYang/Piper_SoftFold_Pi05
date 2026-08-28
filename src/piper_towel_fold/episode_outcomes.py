from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_OUTCOMES = frozenset({"success", "failure", "unknown"})
DEFAULT_OUTCOME_FILTER = "exclude-failures"
OUTCOME_FILTER_CHOICES = frozenset(
    {"exclude-failures", "success-only", "failures-only", "none", "all"}
)


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def outcome_jsonl_path(dataset_root: Path) -> Path:
    return dataset_root / "episode_outcomes.jsonl"


def read_episode_count(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = _read_json(info_path)
        if isinstance(info, dict) and "total_episodes" in info:
            return int(info["total_episodes"])

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        return sum(1 for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip())

    raise FileNotFoundError(
        f"Could not determine episode count under {dataset_root}. "
        "Expected meta/info.json with total_episodes or meta/episodes.jsonl."
    )


def load_outcome_records(dataset_root: Path) -> list[dict[str, Any]]:
    path = outcome_jsonl_path(dataset_root)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid outcome record on line {line_number} of {path}.")
        records.append(record)
    return records


def align_outcomes_to_episodes(
    records: list[dict[str, Any]],
    num_episodes: int,
) -> dict[int, str]:
    """Map episode_index -> outcome.

    Alignment rules (derived from record_episode.py):
    1. Each recording session appends exactly one jsonl line after save_episode().
    2. When episode_index is present in a record, use it directly.
    3. Otherwise, line i (0-based among records without episode_index) maps to episode i.
    """
    outcomes: dict[int, str] = {}
    line_order_records: list[tuple[int, dict[str, Any]]] = []

    for line_index, record in enumerate(records):
        outcome = str(record.get("outcome", "")).strip().lower()
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Unsupported outcome '{outcome}' in episode_outcomes.jsonl.")

        if "episode_index" in record:
            episode_index = int(record["episode_index"])
            outcomes[episode_index] = outcome
        else:
            line_order_records.append((line_index, record))

    for line_index, record in line_order_records:
        episode_index = line_index
        if episode_index in outcomes:
            raise ValueError(
                "episode_outcomes.jsonl has conflicting labels for "
                f"episode {episode_index}. Add explicit episode_index fields."
            )
        outcomes[episode_index] = str(record["outcome"]).strip().lower()

    invalid_indices = sorted(index for index in outcomes if index < 0 or index >= num_episodes)
    if invalid_indices:
        raise ValueError(
            "episode_outcomes.jsonl references episode indices outside the dataset: "
            f"{invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''} "
            f"(dataset has {num_episodes} episodes)."
        )

    return outcomes


def summarize_outcomes(
    dataset_root: Path,
    num_episodes: int | None = None,
) -> dict[str, Any]:
    if num_episodes is None:
        num_episodes = read_episode_count(dataset_root)

    records = load_outcome_records(dataset_root)
    aligned = align_outcomes_to_episodes(records, num_episodes)
    unlabeled = [index for index in range(num_episodes) if index not in aligned]

    counts = {"success": 0, "failure": 0, "unknown": 0, "unlabeled": len(unlabeled)}
    for outcome in aligned.values():
        counts[outcome] += 1

    return {
        "dataset_root": str(dataset_root),
        "num_episodes": num_episodes,
        "num_records": len(records),
        "counts": counts,
        "aligned": aligned,
        "unlabeled": unlabeled,
    }


def select_episodes_for_training(
    dataset_root: Path,
    *,
    outcome_filter: str = DEFAULT_OUTCOME_FILTER,
    include_unknown: bool = False,
    exclude_unlabeled: bool = True,
    manual_episodes: list[int] | None = None,
    num_episodes: int | None = None,
) -> list[int] | None:
    """Return episode indices for lerobot-train, or None to use all episodes."""
    if manual_episodes is not None:
        return sorted({int(index) for index in manual_episodes})

    normalized_filter = outcome_filter.strip().lower()
    if normalized_filter in {"none", "all", ""}:
        return None

    if normalized_filter not in OUTCOME_FILTER_CHOICES:
        raise ValueError(
            f"Unsupported outcome_filter '{outcome_filter}'. "
            f"Choose from: {', '.join(sorted(OUTCOME_FILTER_CHOICES))}."
        )

    if num_episodes is None:
        num_episodes = read_episode_count(dataset_root)

    records = load_outcome_records(dataset_root)
    if not records:
        raise FileNotFoundError(
            f"No labels found at {outcome_jsonl_path(dataset_root)}. "
            "Record episodes with --prompt-outcome or set training.outcome_filter to 'none'."
        )

    aligned = align_outcomes_to_episodes(records, num_episodes)
    selected: list[int] = []

    for episode_index in range(num_episodes):
        outcome = aligned.get(episode_index)
        if outcome is None:
            if exclude_unlabeled:
                continue
            selected.append(episode_index)
            continue

        if normalized_filter == "success-only":
            if outcome == "success":
                selected.append(episode_index)
        elif normalized_filter == "failures-only":
            if outcome == "failure":
                selected.append(episode_index)
        elif normalized_filter == "exclude-failures":
            if outcome == "success":
                selected.append(episode_index)
            elif outcome == "unknown" and include_unknown:
                selected.append(episode_index)
        else:
            raise ValueError(f"Unhandled outcome_filter: {normalized_filter}")

    if not selected:
        raise ValueError(
            f"No episodes selected for training with outcome_filter='{normalized_filter}'. "
            f"Dataset has {num_episodes} episodes and {len(records)} outcome records."
        )

    return selected


def resolve_training_episodes(
    dataset_root: Path,
    training: dict[str, Any],
) -> list[int] | None:
    """Resolve episode list from training config and environment-style overrides."""
    import os

    manual_episodes = training.get("episodes")
    if manual_episodes is None:
        episodes_env = os.environ.get("EPISODES", "").strip()
        if episodes_env:
            manual_episodes = json.loads(episodes_env)
    if manual_episodes is not None:
        return select_episodes_for_training(dataset_root, manual_episodes=list(manual_episodes))

    outcome_filter = training.get("outcome_filter")
    if outcome_filter is None:
        outcome_filter = os.environ.get("OUTCOME_FILTER", DEFAULT_OUTCOME_FILTER)

    legacy_exclude = os.environ.get("EXCLUDE_FAILURES", "").strip().lower()
    if legacy_exclude in {"1", "true", "yes"} and outcome_filter == DEFAULT_OUTCOME_FILTER:
        outcome_filter = "exclude-failures"

    include_unknown = training.get("include_unknown")
    if include_unknown is None:
        include_unknown = os.environ.get("INCLUDE_UNKNOWN", "").strip().lower() in {"1", "true", "yes"}

    exclude_unlabeled = training.get("exclude_unlabeled")
    if exclude_unlabeled is None:
        exclude_unlabeled = os.environ.get("EXCLUDE_UNLABELED", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }

    return select_episodes_for_training(
        dataset_root,
        outcome_filter=str(outcome_filter),
        include_unknown=bool(include_unknown),
        exclude_unlabeled=bool(exclude_unlabeled),
    )
