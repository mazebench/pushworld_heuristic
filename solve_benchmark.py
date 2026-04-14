#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solution_video import render_solution_video, render_solution_video_from_files


REPO_ROOT = Path(__file__).resolve().parent
OG_BENCHMARK_ROOT = REPO_ROOT / "og_benchmark"
PUSHWORLD_CPP_ROOT = REPO_ROOT.parent / "pushworld" / "cpp"
PUSHWORLD_INCLUDE = PUSHWORLD_CPP_ROOT / "include"
PUSHWORLD_SRC = PUSHWORLD_CPP_ROOT / "src" / "pushworld_puzzle.cc"
DOMAIN_TRANSITION_GRAPH_SRC = (
    PUSHWORLD_CPP_ROOT / "src" / "heuristics" / "domain_transition_graph.cc"
)
RECURSIVE_GRAPH_DISTANCE_SRC = (
    PUSHWORLD_CPP_ROOT / "src" / "heuristics" / "recursive_graph_distance.cc"
)
LOCAL_BUILD_DEPS = (
    REPO_ROOT / "boost" / "functional" / "hash.hpp",
    REPO_ROOT / "boost" / "algorithm" / "string.hpp",
)
ADVANCED_SOLVER_MODES = (
    "primitive-rgd",
    "macro-rgd",
    "macro-rgd-deep",
    "macro-portfolio",
    "macro-portfolio-safe",
)
RECURSIVE_SOLVER_MODES = (
    "primitive-rgd",
    "macro-rgd",
    "macro-rgd-deep",
    "macro-recursive",
    "macro-recursive-safe",
)
ADVANCED_MODE_TIMEOUT_GRACE_SECONDS = 0.25


@dataclass(frozen=True)
class SolverLayout:
    solver_name: str
    folder_name: str
    local_source_name: str
    implementation_source: Path
    binary_name: str
    extra_compile_sources: tuple[Path, ...] = ()
    run_kind: str = "json"
    default_time_limit: float = 5.0
    modes: tuple[str, ...] = ()

    @property
    def root(self) -> Path:
        return REPO_ROOT / self.folder_name

    @property
    def local_source(self) -> Path:
        return self.root / self.local_source_name

    @property
    def binary(self) -> Path:
        return self.root / self.binary_name

    @property
    def solutions_root(self) -> Path:
        return self.root / "solutions"

    @property
    def summary_path(self) -> Path:
        return self.solutions_root / "summary.json"


SOLVER_LAYOUTS = {
    "bfs": SolverLayout(
        solver_name="bfs",
        folder_name="heuristic_bfs",
        local_source_name="pwp_bfs_solver.cpp",
        implementation_source=REPO_ROOT / "pwp_bfs_solver.cpp",
        binary_name="pwp_bfs_solver",
    ),
    "astar": SolverLayout(
        solver_name="astar",
        folder_name="heuristic_a_star",
        local_source_name="pwp_astar_solver.cpp",
        implementation_source=REPO_ROOT / "heuristic_a_star" / "pwp_astar_solver.cpp",
        binary_name="pwp_astar_solver",
    ),
    "astar_boxes_are_walls": SolverLayout(
        solver_name="astar_boxes_are_walls",
        folder_name="heuristic_a_star_boxes_are_walls",
        local_source_name="pwp_astar_solver.cpp",
        implementation_source=REPO_ROOT
        / "heuristic_a_star_boxes_are_walls"
        / "pwp_astar_solver.cpp",
        binary_name="pwp_astar_solver",
    ),
    "astar_blockers": SolverLayout(
        solver_name="astar_blockers",
        folder_name="heuristic_a_start_blocker",
        local_source_name="pwp_astar_blockers_solver.cpp",
        implementation_source=REPO_ROOT / "pwp_astar_blockers_solver.cpp",
        binary_name="pwp_astar_blockers_solver",
    ),
    "fixed_blockers": SolverLayout(
        solver_name="fixed_blockers",
        folder_name="heuristic_fixed_blockers",
        local_source_name="pwp_astar_fixed_blockers_solver.cpp",
        implementation_source=REPO_ROOT / "pwp_astar_fixed_blockers_solver.cpp",
        binary_name="pwp_astar_fixed_blockers_solver",
    ),
    "many": SolverLayout(
        solver_name="many",
        folder_name="heuristic_many",
        local_source_name="advanced_pushworld_solver.cpp",
        implementation_source=REPO_ROOT / "heuristic_many" / "advanced_pushworld_solver.cpp",
        binary_name="advanced_pushworld_solver",
        extra_compile_sources=(
            DOMAIN_TRANSITION_GRAPH_SRC,
            RECURSIVE_GRAPH_DISTANCE_SRC,
        ),
        run_kind="advanced_many",
        default_time_limit=30.0,
        modes=ADVANCED_SOLVER_MODES,
    ),
    "recursive": SolverLayout(
        solver_name="recursive",
        folder_name="heuristic_recursive",
        local_source_name="advanced_pushworld_recursive_solver.cpp",
        implementation_source=REPO_ROOT / "heuristic_recursive" / "advanced_pushworld_recursive_solver.cpp",
        binary_name="advanced_pushworld_recursive_solver",
        extra_compile_sources=(
            DOMAIN_TRANSITION_GRAPH_SRC,
            RECURSIVE_GRAPH_DISTANCE_SRC,
        ),
        run_kind="advanced_many",
        default_time_limit=30.0,
        modes=RECURSIVE_SOLVER_MODES,
    ),
    "recursive_bidirectional": SolverLayout(
        solver_name="recursive_bidirectional",
        folder_name="heuristic_recursive_bidirectional",
        local_source_name="advanced_pushworld_recursive_bidirectional_solver.cpp",
        implementation_source=REPO_ROOT
        / "heuristic_recursive_bidirectional"
        / "advanced_pushworld_recursive_bidirectional_solver.cpp",
        binary_name="advanced_pushworld_recursive_bidirectional_solver",
        extra_compile_sources=(
            DOMAIN_TRANSITION_GRAPH_SRC,
            RECURSIVE_GRAPH_DISTANCE_SRC,
        ),
        run_kind="advanced_many",
        default_time_limit=30.0,
        modes=RECURSIVE_SOLVER_MODES,
    ),
    "recursive_bidirectional_better": SolverLayout(
        solver_name="recursive_bidirectional_better",
        folder_name="heuristic_recursive_bidirectional_better",
        local_source_name="advanced_pushworld_recursive_bidirectional_better_solver.cpp",
        implementation_source=REPO_ROOT
        / "heuristic_recursive_bidirectional_better"
        / "advanced_pushworld_recursive_bidirectional_better_solver.cpp",
        binary_name="advanced_pushworld_recursive_bidirectional_better_solver",
        extra_compile_sources=(
            DOMAIN_TRANSITION_GRAPH_SRC,
            RECURSIVE_GRAPH_DISTANCE_SRC,
        ),
        run_kind="advanced_many",
        default_time_limit=30.0,
        modes=RECURSIVE_SOLVER_MODES,
    ),
}


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
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    write_text_atomic(path, text)


def discover_levels(root: Path) -> list[LevelJob]:
    jobs: list[LevelJob] = []
    for source_path in sorted(root.rglob("*.pwp")):
        rel_path = source_path.relative_to(root)
        level_dir = rel_path.parts[0]
        if not level_dir.startswith("level"):
            raise ValueError(f"unexpected level folder {level_dir!r}")
        difficulty = int(level_dir.removeprefix("level"))
        jobs.append(
            LevelJob(
                source_path=source_path,
                difficulty=difficulty,
                level_name=source_path.stem,
                relative_stem=rel_path.with_suffix(""),
            )
        )
    return jobs


def solution_path_for(layout: SolverLayout, job: LevelJob) -> Path:
    return layout.solutions_root / job.relative_stem.with_suffix(".txt")


def solution_video_path_for(layout: SolverLayout, job: LevelJob) -> Path:
    return layout.root / "solutions_video" / job.relative_stem.with_suffix(".mp4")


def render_cached_solution_video(layout: SolverLayout, job: LevelJob) -> Path:
    solution_path = solution_path_for(layout, job)
    video_path = solution_video_path_for(layout, job)
    render_solution_video_from_files(job.source_path, solution_path, video_path)
    return video_path


def ensure_solver_binary(layout: SolverLayout) -> Path:
    if not PUSHWORLD_INCLUDE.exists() or not PUSHWORLD_SRC.exists():
        raise FileNotFoundError(
            "Missing sibling pushworld repo pieces. Expected "
            f"{PUSHWORLD_INCLUDE} and {PUSHWORLD_SRC}."
        )
    if not layout.local_source.exists():
        raise FileNotFoundError(
            f"missing wrapper source for {layout.solver_name}: {layout.local_source}"
        )
    for extra_source in layout.extra_compile_sources:
        if not extra_source.exists():
            raise FileNotFoundError(
                f"missing extra source for {layout.solver_name}: {extra_source}"
            )

    needs_build = not layout.binary.exists()
    if not needs_build:
        binary_mtime = layout.binary.stat().st_mtime
        dependency_paths = list(LOCAL_BUILD_DEPS) + [layout.local_source, PUSHWORLD_SRC]
        if layout.implementation_source != layout.local_source:
            dependency_paths.append(layout.implementation_source)
        dependency_paths.extend(layout.extra_compile_sources)
        needs_build = any(binary_mtime < path.stat().st_mtime for path in dependency_paths)
    if not needs_build:
        return layout.binary

    layout.root.mkdir(parents=True, exist_ok=True)
    command = [
        "c++",
        "-std=c++20",
        "-O3",
        str(layout.local_source),
        str(PUSHWORLD_SRC),
        *[str(source) for source in layout.extra_compile_sources],
        "-I",
        str(REPO_ROOT),
        "-I",
        str(PUSHWORLD_INCLUDE),
        "-o",
        str(layout.binary),
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
        raise RuntimeError(f"failed to build {layout.binary.name}: {detail}")
    return layout.binary


def load_existing_summary(path: Path, default_solver_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("solver", default_solver_name)
        entries.append(normalized)
    return entries


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


def run_json_level(binary: Path, job: LevelJob, time_limit_seconds: float) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), str(job.source_path), f"{time_limit_seconds:.6f}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"{binary.name} produced no output for {job.source_path}"
        )
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


def parse_metric_fields(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for token in text.replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            metrics[key] = value
    return metrics


def metric_flag(metrics: dict[str, str], key: str) -> bool:
    return metrics.get(key, "").lower() in {"1", "true", "yes"}


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_advanced_mode_payload(
    mode: str,
    returncode: int,
    stdout: str,
    stderr: str,
    measured_duration: float,
) -> dict[str, Any]:
    trimmed_stdout = stdout.strip()
    trimmed_stderr = stderr.strip()
    metrics = parse_metric_fields(trimmed_stderr)
    duration = float(metrics.get("elapsed_seconds", measured_duration))
    nodes_expanded = int(metrics.get("expansions", 0))
    nodes_generated = int(metrics.get("generated", 0))
    reopened = int(metrics.get("reopened", 0))
    timed_out = metric_flag(metrics, "timed_out")

    if returncode != 0:
        detail = trimmed_stderr or trimmed_stdout or f"{mode} exited with code {returncode}"
        return {
            "ok": False,
            "solved": False,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
            "reopened": reopened,
            "mode": mode,
            "error": detail,
        }

    if not trimmed_stdout:
        return {
            "ok": False,
            "solved": False,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
            "reopened": reopened,
            "mode": mode,
            "error": f"{mode} produced no output",
        }

    solved = trimmed_stdout != "NO SOLUTION"
    payload: dict[str, Any] = {
        "ok": True,
        "solved": solved,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "reopened": reopened,
        "mode": mode,
    }
    if solved:
        payload["solution"] = trimmed_stdout
    return payload


def run_advanced_mode(
    binary: Path,
    mode: str,
    job: LevelJob,
    time_limit_seconds: float,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {"cancelled": True, "mode": mode}

    command = [
        str(binary),
        mode,
        str(job.source_path),
        f"{time_limit_seconds:.6f}",
    ]
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while True:
            if stop_event.is_set():
                stop_process(process)
                stdout, stderr = process.communicate()
                return {"cancelled": True, "mode": mode}
            elapsed = time.perf_counter() - start
            if elapsed > time_limit_seconds + ADVANCED_MODE_TIMEOUT_GRACE_SECONDS:
                stop_process(process)
                stdout, stderr = process.communicate()
                metrics = parse_metric_fields(stderr.strip())
                return {
                    "ok": True,
                    "solved": False,
                    "timed_out": True,
                    "duration_seconds": float(metrics.get("elapsed_seconds", elapsed)),
                    "nodes_expanded": int(metrics.get("expansions", 0)),
                    "nodes_generated": int(metrics.get("generated", 0)),
                    "reopened": int(metrics.get("reopened", 0)),
                    "mode": mode,
                }
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                break
            time.sleep(0.05)
    finally:
        if stop_event.is_set():
            stop_process(process)

    measured_duration = time.perf_counter() - start
    return parse_advanced_mode_payload(
        mode,
        process.returncode or 0,
        stdout,
        stderr,
        measured_duration,
    )


def run_advanced_many_level(
    layout: SolverLayout,
    binary: Path,
    job: LevelJob,
    time_limit_seconds: float,
) -> dict[str, Any]:
    stop_event = threading.Event()
    results: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None

    with ThreadPoolExecutor(max_workers=len(layout.modes)) as executor:
        future_to_mode = {
            executor.submit(
                run_advanced_mode,
                binary,
                mode,
                job,
                time_limit_seconds,
                stop_event,
            ): mode
            for mode in layout.modes
        }

        for future in as_completed(future_to_mode):
            mode = future_to_mode[future]
            try:
                result = future.result()
            except Exception as exc:
                result = build_error_payload(exc)
                result["mode"] = mode

            if result.get("cancelled", False):
                continue

            results.append(result)
            if winner is None and result.get("solved", False):
                winner = result
                stop_event.set()

    if winner is not None:
        winner = dict(winner)
        winner["solver"] = layout.solver_name
        return winner

    if not results:
        return build_error_payload(
            RuntimeError(f"{layout.solver_name} produced no mode results for {job.source_path}")
        )

    overall_ok = any(result.get("ok", False) for result in results)
    combined: dict[str, Any] = {
        "ok": overall_ok,
        "solved": False,
        "timed_out": any(result.get("timed_out", False) for result in results),
        "duration_seconds": max(float(result.get("duration_seconds", 0.0)) for result in results),
        "nodes_expanded": sum(int(result.get("nodes_expanded", 0)) for result in results),
        "nodes_generated": sum(int(result.get("nodes_generated", 0)) for result in results),
    }
    if not overall_ok:
        errors = [str(result.get("error", "")) for result in results if result.get("error")]
        combined["error"] = " | ".join(errors) or f"all {layout.solver_name} modes failed"
    return combined


def run_level(
    layout: SolverLayout,
    binary: Path,
    job: LevelJob,
    time_limit_seconds: float,
) -> dict[str, Any]:
    if layout.run_kind == "json":
        return run_json_level(binary, job, time_limit_seconds)
    if layout.run_kind == "advanced_many":
        return run_advanced_many_level(layout, binary, job, time_limit_seconds)
    raise ValueError(f"unsupported run kind {layout.run_kind!r}")


def build_summary_entry(
    job: LevelJob,
    payload: dict[str, Any],
    time_limit_seconds: float,
    solver_name: str,
) -> dict[str, Any]:
    solved = bool(payload.get("solved", False))
    timed_out = bool(payload.get("timed_out", False))
    duration = float(payload.get("duration_seconds", 0.0))
    nodes_searched = int(payload.get("nodes_expanded", 0))
    entry: dict[str, Any] = {
        "level_difficulty": job.difficulty,
        "level_name": job.level_name,
        "solved": solved,
        "duration": duration,
        "nodes_searched": nodes_searched,
        "time_limit_seconds": time_limit_seconds,
        "solver": solver_name,
    }
    if timed_out:
        entry["status"] = "timed_out"
    elif solved:
        entry["status"] = "solved"
    elif payload.get("ok", True):
        entry["status"] = "no_solution"
    else:
        entry["status"] = "error"

    if "nodes_generated" in payload:
        entry["nodes_generated"] = int(payload["nodes_generated"])
    if solved:
        entry["solution_length"] = len(str(payload.get("solution", "")))
    if "mode" in payload:
        entry["mode"] = str(payload["mode"])
    if "reopened" in payload:
        entry["reopened"] = int(payload["reopened"])
    if "error" in payload:
        entry["error"] = str(payload["error"])
    return entry


def build_error_payload(error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "solved": False,
        "timed_out": False,
        "duration_seconds": 0.0,
        "nodes_expanded": 0,
        "nodes_generated": 0,
        "error": str(error),
    }


def should_skip(
    job: LevelJob,
    entries: list[dict[str, Any]],
    layout: SolverLayout,
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
    if not isinstance(prior, dict):
        return False

    solution_path = solution_path_for(layout, job)
    if bool(prior.get("solved", False)) and solution_path.exists():
        return True

    prior_limit = prior.get("time_limit_seconds")
    if (
        prior.get("status") in {"timed_out", "no_solution", "error"}
        and isinstance(prior_limit, (int, float))
        and float(prior_limit) >= time_limit_seconds
    ):
        return True

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve PushWorld benchmark levels and store solutions/summary files."
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Per-level search cap in seconds. Defaults to 5, or 30 for solvers 'many' and 'recursive'.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip levels already solved for this heuristic. Default: on.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Only run one difficulty folder, e.g. 1 or 4.",
    )
    parser.add_argument(
        "--solver",
        choices=tuple(SOLVER_LAYOUTS.keys()),
        default="bfs",
        help="Search backend to build and run. Default: bfs.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of puzzles to solve concurrently. Default: 1. Ignored for solvers 'many' and 'recursive'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")

    layout = SOLVER_LAYOUTS[args.solver]
    time_limit_seconds = (
        layout.default_time_limit if args.time_limit is None else args.time_limit
    )
    effective_threads = args.threads
    if layout.run_kind == "advanced_many":
        effective_threads = 1
        if args.threads != 1:
            print(
                f"Note: solver '{layout.solver_name}' already races 5 internal modes per puzzle; "
                f"ignoring --threads={args.threads} and running one puzzle at a time.",
                flush=True,
            )
    solver_binary = ensure_solver_binary(layout)
    jobs = discover_levels(OG_BENCHMARK_ROOT)
    if args.level is not None:
        jobs = [job for job in jobs if job.difficulty == args.level]

    entries = load_existing_summary(layout.summary_path, layout.solver_name)
    layout.solutions_root.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    completed_count = 0
    rendered_count = 0
    skipped_count = 0
    pending_jobs: list[tuple[int, LevelJob]] = []

    for index, job in enumerate(jobs, start=1):
        solution_path = solution_path_for(layout, job)
        video_path = solution_video_path_for(layout, job)
        if args.resume and solution_path.exists():
            if video_path.exists():
                skipped_count += 1
                print(
                    f"[{index}/{total}] skip {layout.folder_name}/level{job.difficulty}/{job.level_name} "
                    "(solution and video already exist)",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{total}] render {layout.folder_name}/level{job.difficulty}/{job.level_name} "
                    "from cached solution",
                    flush=True,
                )
                render_cached_solution_video(layout, job)
                rendered_count += 1
            continue

        if should_skip(job, entries, layout, args.resume, time_limit_seconds):
            skipped_count += 1
            print(
                f"[{index}/{total}] skip {layout.folder_name}/level{job.difficulty}/{job.level_name}",
                flush=True,
            )
            continue

        print(
            f"[{index}/{total}] queue {layout.folder_name}/level{job.difficulty}/{job.level_name}",
            flush=True,
        )
        pending_jobs.append((index, job))

    if pending_jobs:
        if layout.run_kind == "advanced_many":
            print(
                f"Starting {len(pending_jobs)} puzzle(s) sequentially with "
                f"{len(layout.modes)} internal mode workers per puzzle.",
                flush=True,
            )
        else:
            print(
                f"Starting {len(pending_jobs)} puzzle(s) with {effective_threads} worker(s).",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=effective_threads) as executor:
        future_to_job = {
            executor.submit(run_level, layout, solver_binary, job, time_limit_seconds): (index, job)
            for index, job in pending_jobs
        }

        for future in as_completed(future_to_job):
            index, job = future_to_job[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = build_error_payload(exc)

            solution_path = solution_path_for(layout, job)
            solution_video_path = solution_video_path_for(layout, job)
            if payload.get("solved", False):
                solution_text = str(payload.get("solution", "")) + "\n"
                write_text_atomic(solution_path, solution_text)
                render_solution_video(
                    puzzle_path=job.source_path,
                    solution_text=solution_text,
                    video_path=solution_video_path,
                )
            else:
                if solution_path.exists():
                    solution_path.unlink()
                if solution_video_path.exists():
                    solution_video_path.unlink()

            entry = build_summary_entry(job, payload, time_limit_seconds, layout.solver_name)
            update_summary_entry(entries, entry)
            write_json_atomic(layout.summary_path, entries)

            completed_count += 1
            print(
                f"[{index}/{total}] done {layout.folder_name}/level{job.difficulty}/{job.level_name} "
                f"-> {entry['status']} in {entry['duration']:.3f}s, "
                f"nodes={entry['nodes_searched']}",
                flush=True,
            )

    print(
        f"Done. processed={completed_count} rendered={rendered_count} skipped={skipped_count} "
        f"summary={layout.summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
