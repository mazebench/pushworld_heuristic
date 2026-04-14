#!/usr/bin/env python3

from __future__ import annotations

import string
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPO_ROOT / "og_benchmark"
TARGET_ROOT = REPO_ROOT / "benchmark"
GATE_SUFFIX = "|"


def index_to_letter(index: int) -> str:
    if index < 0 or index >= len(string.ascii_lowercase):
        raise ValueError(f"box/goal index {index} is out of range for single-letter output")
    return string.ascii_lowercase[index]


def translate_part(part: str, occupant: str, terrain: str) -> tuple[str, str]:
    if part == ".":
        return occupant, terrain
    if part == "A":
        if occupant != ".":
            raise ValueError(f"cell already has occupant {occupant!r}")
        return "P", terrain
    if part == "W":
        if occupant != "." or terrain != ".":
            raise ValueError("wall token cannot be combined with other cell content")
        return "#", "."
    if part == "AW":
        if terrain != ".":
            raise ValueError(f"cell already has terrain {terrain!r}")
        return occupant, GATE_SUFFIX
    if part.startswith("M") and part[1:].isdigit():
        if occupant != ".":
            raise ValueError(f"cell already has occupant {occupant!r}")
        return index_to_letter(int(part[1:])), terrain
    if part.startswith("G") and part[1:].isdigit():
        if terrain != ".":
            raise ValueError(f"cell already has terrain {terrain!r}")
        return occupant, index_to_letter(int(part[1:]))
    raise ValueError(f"unrecognized token part {part!r}")


def translate_token(token: str) -> str:
    occupant = "."
    terrain = "."
    for part in token.split("+"):
        occupant, terrain = translate_part(part, occupant, terrain)
    return occupant + terrain


def translate_file(source_path: Path, target_path: Path) -> None:
    rows: list[str] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        rows.append(" ".join(translate_token(token) for token in tokens))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    source_files = sorted(SOURCE_ROOT.rglob("*.pwp"))
    translated = 0
    gate_files = 0
    for source_path in source_files:
        rel_path = source_path.relative_to(SOURCE_ROOT)
        target_path = (TARGET_ROOT / rel_path).with_suffix(".txt")
        text = source_path.read_text(encoding="utf-8")
        if "AW" in text:
            gate_files += 1
        translate_file(source_path, target_path)
        translated += 1
    print(
        f"Translated {translated} level files into {TARGET_ROOT} "
        f"({gate_files} files include gates encoded as .{GATE_SUFFIX} / x{GATE_SUFFIX})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
