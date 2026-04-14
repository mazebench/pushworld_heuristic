#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solution_video import PushWorldPuzzleVideo, normalize_plan_text, render_puzzle_png


REPO_ROOT = Path(__file__).resolve().parent
OG_BENCHMARK_ROOT = REPO_ROOT / "og_benchmark"
SPREAD_BENCHMARK_ROOT = REPO_ROOT / "spread_benchmark"
PUSHWORLD_CPP_ROOT = REPO_ROOT.parent / "pushworld" / "cpp"
PUSHWORLD_INCLUDE = PUSHWORLD_CPP_ROOT / "include"
PUSHWORLD_SRC = PUSHWORLD_CPP_ROOT / "src" / "pushworld_puzzle.cc"
DOMAIN_TRANSITION_GRAPH_SRC = (
    PUSHWORLD_CPP_ROOT / "src" / "heuristics" / "domain_transition_graph.cc"
)
LOCAL_BUILD_DEPS = (
    REPO_ROOT / "boost" / "functional" / "hash.hpp",
    REPO_ROOT / "boost" / "algorithm" / "string.hpp",
)
SPREAD_SOURCE = (
    REPO_ROOT
    / "heuristic_scramble_and_spread"
    / "scramble_and_spread_solver.cpp"
)
SPREAD_BINARY = (
    REPO_ROOT
    / "heuristic_scramble_and_spread"
    / "scramble_and_spread_solver"
)
SUMMARY_PATH = SPREAD_BENCHMARK_ROOT / "summary.json"


@dataclass(frozen=True)
class LevelJob:
    source_path: Path
    difficulty: int
    level_name: str
    relative_stem: Path


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def summary_index(entries: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    index: dict[tuple[int, str], int] = {}
    for position, entry in enumerate(entries):
        difficulty = entry.get("level_difficulty")
        level_name = entry.get("level_name")
        if isinstance(difficulty, int) and isinstance(level_name, str):
            index[(difficulty, level_name)] = position
    return index


def update_summary_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    index = summary_index(entries)
    key = (entry["level_difficulty"], entry["level_name"])
    if key in index:
        entries[index[key]] = entry
    else:
        entries.append(entry)
    entries.sort(key=lambda item: (item["level_difficulty"], item["level_name"].lower()))


def discover_levels(root: Path) -> list[LevelJob]:
    jobs: list[LevelJob] = []
    for source_path in sorted(root.rglob("*.pwp")):
        rel_path = source_path.relative_to(root)
        level_dir = rel_path.parts[0]
        if not level_dir.startswith("level"):
            raise ValueError(f"unexpected level folder {level_dir!r}")
        jobs.append(
            LevelJob(
                source_path=source_path,
                difficulty=int(level_dir.removeprefix("level")),
                level_name=source_path.stem,
                relative_stem=rel_path.with_suffix(""),
            )
        )
    return jobs


def spread_puzzle_path(job: LevelJob) -> Path:
    return SPREAD_BENCHMARK_ROOT / job.relative_stem.with_suffix(".pwp")


def spread_plan_path(job: LevelJob) -> Path:
    return SPREAD_BENCHMARK_ROOT / job.relative_stem.with_suffix(".txt")


def spread_png_path(job: LevelJob) -> Path:
    return SPREAD_BENCHMARK_ROOT / job.relative_stem.with_suffix(".png")


def ensure_solver_binary() -> Path:
    if not PUSHWORLD_INCLUDE.exists() or not PUSHWORLD_SRC.exists():
        raise FileNotFoundError(
            "Missing sibling pushworld repo pieces. Expected "
            f"{PUSHWORLD_INCLUDE} and {PUSHWORLD_SRC}."
        )
    if not SPREAD_SOURCE.exists():
        raise FileNotFoundError(f"missing spread solver source: {SPREAD_SOURCE}")
    if not DOMAIN_TRANSITION_GRAPH_SRC.exists():
        raise FileNotFoundError(
            f"missing domain transition graph source: {DOMAIN_TRANSITION_GRAPH_SRC}"
        )

    needs_build = not SPREAD_BINARY.exists()
    if not needs_build:
        binary_mtime = SPREAD_BINARY.stat().st_mtime
        dependency_paths = [
            *LOCAL_BUILD_DEPS,
            SPREAD_SOURCE,
            PUSHWORLD_SRC,
            DOMAIN_TRANSITION_GRAPH_SRC,
        ]
        needs_build = any(binary_mtime < path.stat().st_mtime for path in dependency_paths)
    if not needs_build:
        return SPREAD_BINARY

    SPREAD_BINARY.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "c++",
        "-std=c++20",
        "-O3",
        str(SPREAD_SOURCE),
        str(PUSHWORLD_SRC),
        str(DOMAIN_TRANSITION_GRAPH_SRC),
        "-I",
        str(REPO_ROOT),
        "-I",
        str(PUSHWORLD_INCLUDE),
        "-o",
        str(SPREAD_BINARY),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown compiler failure"
        raise RuntimeError(f"failed to build {SPREAD_BINARY.name}: {detail}")
    return SPREAD_BINARY


def run_level(binary: Path, job: LevelJob, time_limit_seconds: float) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), str(job.source_path), f"{time_limit_seconds:.6f}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{binary.name} produced no output for {job.source_path}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{binary.name} produced invalid JSON for {job.source_path}: {stdout}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{binary.name} produced a non-object JSON payload")
    if completed.returncode != 0 and payload.get("ok", True):
        raise RuntimeError(
            f"{binary.name} exited with code {completed.returncode} for {job.source_path}"
        )
    return payload


def build_error_payload(error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "solved": False,
        "timed_out": False,
        "apex_reached": False,
        "duration_seconds": 0.0,
        "nodes_expanded": 0,
        "nodes_generated": 0,
        "plan": "",
        "error": str(error),
    }


def materialize_level(job: LevelJob, payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    plan_text = normalize_plan_text(str(payload.get("plan", "")))
    puzzle = PushWorldPuzzleVideo(job.source_path)
    final_state = puzzle.apply_plan(plan_text)

    puzzle_path = spread_puzzle_path(job)
    plan_path = spread_plan_path(job)
    png_path = spread_png_path(job)

    write_text_atomic(puzzle_path, puzzle.serialize_state(final_state))
    write_text_atomic(plan_path, plan_text + "\n")
    puzzle.render_png(png_path, state=final_state)
    return puzzle_path, plan_path, png_path


def build_summary_entry(
    job: LevelJob,
    payload: dict[str, Any],
    time_limit_seconds: float,
    output_paths: tuple[Path, Path, Path] | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "level_difficulty": job.difficulty,
        "level_name": job.level_name,
        "solved": bool(payload.get("solved", False)),
        "timed_out": bool(payload.get("timed_out", False)),
        "apex_reached": bool(payload.get("apex_reached", False)),
        "duration": float(payload.get("duration_seconds", 0.0)),
        "nodes_searched": int(payload.get("nodes_expanded", 0)),
        "nodes_generated": int(payload.get("nodes_generated", 0)),
        "time_limit_seconds": time_limit_seconds,
        "plan_length": len(str(payload.get("plan", ""))),
    }
    if not payload.get("ok", True):
        entry["status"] = "error"
        entry["error"] = str(payload.get("error", "unknown error"))
        return entry

    entry["status"] = "solved" if entry["solved"] else "spread"
    for field in (
        "best_score",
        "deadlocked_target_boxes",
        "non_target_boxes_on_goals",
        "targets_solved",
        "goal_distance_sum",
        "pair_distance_min",
        "pair_distance_sum",
        "mobile_boxes",
        "pinned_boxes",
        "mobility_sum",
    ):
        if field in payload:
            entry[field] = payload[field]

    if output_paths is not None:
        puzzle_path, plan_path, png_path = output_paths
        entry["output_puzzle_path"] = str(puzzle_path.relative_to(REPO_ROOT))
        entry["output_plan_path"] = str(plan_path.relative_to(REPO_ROOT))
        entry["output_png_path"] = str(png_path.relative_to(REPO_ROOT))
    return entry


def should_skip(
    job: LevelJob,
    entries: list[dict[str, Any]],
    resume: bool,
    time_limit_seconds: float,
) -> bool:
    if not resume:
        return False
    entry_map = summary_index(entries)
    entry_index = entry_map.get((job.difficulty, job.level_name))
    if entry_index is None:
        return False

    prior = entries[entry_index]
    if prior.get("status") == "error":
        return False
    prior_limit = prior.get("time_limit_seconds")
    if not isinstance(prior_limit, (int, float)) or float(prior_limit) < time_limit_seconds:
        return False
    return (
        spread_puzzle_path(job).exists()
        and spread_plan_path(job).exists()
        and spread_png_path(job).exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spread-oriented PushWorld benchmark states."
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1.0,
        help="Per-level scramble budget in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip levels that already have spread artifacts for the same or longer time limit.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Only run one difficulty folder, e.g. 1 or 4.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of puzzles to scramble concurrently. Default: 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")

    binary = ensure_solver_binary()
    entries = load_existing_summary(SUMMARY_PATH)
    jobs = discover_levels(OG_BENCHMARK_ROOT)
    if args.level is not None:
        jobs = [job for job in jobs if job.difficulty == args.level]

    total = len(jobs)
    pending_jobs: list[tuple[int, LevelJob]] = []
    skipped_count = 0

    for index, job in enumerate(jobs, start=1):
        if should_skip(job, entries, args.resume, args.time_limit):
            skipped_count += 1
            print(
                f"[{index}/{total}] skip spread_benchmark/level{job.difficulty}/{job.level_name}",
                flush=True,
            )
            continue

        if args.resume and spread_puzzle_path(job).exists() and spread_plan_path(job).exists() and not spread_png_path(job).exists():
            render_puzzle_png(spread_puzzle_path(job), spread_png_path(job))
            skipped_count += 1
            print(
                f"[{index}/{total}] render spread_benchmark/level{job.difficulty}/{job.level_name} png",
                flush=True,
            )
            continue

        pending_jobs.append((index, job))
        print(
            f"[{index}/{total}] queue spread_benchmark/level{job.difficulty}/{job.level_name}",
            flush=True,
        )

    if pending_jobs:
        print(
            f"Starting {len(pending_jobs)} puzzle(s) with {args.threads} worker(s).",
            flush=True,
        )

    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_job = {
            executor.submit(run_level, binary, job, args.time_limit): (index, job)
            for index, job in pending_jobs
        }
        for future in as_completed(future_to_job):
            index, job = future_to_job[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = build_error_payload(exc)

            output_paths: tuple[Path, Path, Path] | None = None
            if payload.get("ok", True):
                try:
                    output_paths = materialize_level(job, payload)
                except Exception as exc:
                    payload = build_error_payload(exc)
                    output_paths = None

            entry = build_summary_entry(job, payload, args.time_limit, output_paths)
            update_summary_entry(entries, entry)
            write_json_atomic(SUMMARY_PATH, entries)
            completed_count += 1

            print(
                f"[{index}/{total}] done spread_benchmark/level{job.difficulty}/{job.level_name} "
                f"-> {entry['status']} in {entry['duration']:.3f}s, "
                f"nodes={entry['nodes_searched']}",
                flush=True,
            )

    print(
        f"Done. processed={completed_count} skipped={skipped_count} summary={SUMMARY_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
