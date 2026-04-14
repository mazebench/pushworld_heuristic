#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from solution_video import render_solution_video_from_files


REPO_ROOT = Path(__file__).resolve().parent
OG_BENCHMARK_ROOT = REPO_ROOT / "og_benchmark"
MASTER_SOLUTIONS_ROOT = REPO_ROOT / "master_solutions"
MASTER_SOLUTIONS_VIDEO_ROOT = REPO_ROOT / "master_solutions_video"
MASTER_SUMMARY_PATH = REPO_ROOT / "master_summary.json"

HEURISTIC_FOLDERS: dict[str, Path] = {
    "bfs": REPO_ROOT / "heuristic_bfs",
    "astar": REPO_ROOT / "heuristic_a_star",
    "astar_boxes_are_walls": REPO_ROOT / "heuristic_a_star_boxes_are_walls",
    "astar_blockers": REPO_ROOT / "heuristic_a_start_blocker",
    "fixed_blockers": REPO_ROOT / "heuristic_fixed_blockers",
    "many": REPO_ROOT / "heuristic_many",
    "recursive": REPO_ROOT / "heuristic_recursive",
    "recursive_bidirectional": REPO_ROOT / "heuristic_recursive_bidirectional",
    "recursive_bidirectional_better": REPO_ROOT / "heuristic_recursive_bidirectional_better",
}


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    write_text_atomic(path, text)


def load_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def clear_existing_artifacts(root: Path, suffix: str) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob(f"*{suffix}")):
        path.unlink()


def solution_path_for(heuristic_root: Path, difficulty: int, level_name: str) -> Path:
    return heuristic_root / "solutions" / f"level{difficulty}" / f"{level_name}.txt"


def solution_video_path_for(heuristic_root: Path, difficulty: int, level_name: str) -> Path:
    return heuristic_root / "solutions_video" / f"level{difficulty}" / f"{level_name}.mp4"


def winner_sort_key(entry: dict[str, Any]) -> tuple[float, int, int, str]:
    duration = float(entry.get("duration", float("inf")))
    solution_length = int(entry.get("solution_length", 10**9))
    nodes = int(entry.get("nodes_searched", 10**18))
    heuristic = str(entry.get("heuristic", ""))
    return (duration, solution_length, nodes, heuristic)


def main() -> int:
    best_by_level: dict[tuple[int, str], dict[str, Any]] = {}

    for heuristic_name, heuristic_root in HEURISTIC_FOLDERS.items():
        summary_path = heuristic_root / "solutions" / "summary.json"
        for entry in load_summary(summary_path):
            if not bool(entry.get("solved", False)):
                continue

            difficulty = entry.get("level_difficulty")
            level_name = entry.get("level_name")
            if not isinstance(difficulty, int) or not isinstance(level_name, str):
                continue

            source_solution_path = solution_path_for(heuristic_root, difficulty, level_name)
            if not source_solution_path.exists():
                continue

            candidate = dict(entry)
            candidate["heuristic"] = heuristic_name
            candidate["source_solution_path"] = str(source_solution_path.relative_to(REPO_ROOT))
            candidate["source_solution_video_path"] = str(
                solution_video_path_for(heuristic_root, difficulty, level_name).relative_to(REPO_ROOT)
            )
            candidate["master_solution_path"] = str(
                (MASTER_SOLUTIONS_ROOT / f"level{difficulty}" / f"{level_name}.txt").relative_to(REPO_ROOT)
            )
            candidate["master_solution_video_path"] = str(
                (MASTER_SOLUTIONS_VIDEO_ROOT / f"level{difficulty}" / f"{level_name}.mp4").relative_to(REPO_ROOT)
            )

            key = (difficulty, level_name)
            incumbent = best_by_level.get(key)
            if incumbent is None or winner_sort_key(candidate) < winner_sort_key(incumbent):
                best_by_level[key] = candidate

    clear_existing_artifacts(MASTER_SOLUTIONS_ROOT, ".txt")
    clear_existing_artifacts(MASTER_SOLUTIONS_VIDEO_ROOT, ".mp4")
    winners = sorted(
        best_by_level.values(),
        key=lambda item: (item["level_difficulty"], item["level_name"].lower()),
    )

    for winner in winners:
        destination = REPO_ROOT / winner["master_solution_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_solution_path = REPO_ROOT / winner["source_solution_path"]
        shutil.copyfile(source_solution_path, destination)

        source_video_path = REPO_ROOT / winner["source_solution_video_path"]
        destination_video = REPO_ROOT / winner["master_solution_video_path"]
        destination_video.parent.mkdir(parents=True, exist_ok=True)
        if source_video_path.exists():
            shutil.copyfile(source_video_path, destination_video)
        else:
            puzzle_path = OG_BENCHMARK_ROOT / f"level{winner['level_difficulty']}" / f"{winner['level_name']}.pwp"
            render_solution_video_from_files(puzzle_path, source_solution_path, destination_video)

    write_json_atomic(MASTER_SUMMARY_PATH, winners)
    print(
        f"Wrote {len(winners)} master solutions to {MASTER_SOLUTIONS_ROOT}, "
        f"master videos to {MASTER_SOLUTIONS_VIDEO_ROOT}, "
        f"and summary to {MASTER_SUMMARY_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
