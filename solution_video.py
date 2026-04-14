#!/usr/bin/env python3

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
from typing import Iterable
import zlib


Point = tuple[int, int]
Color = tuple[int, int, int]

ACTION_DISPLACEMENTS: dict[str, Point] = {
    "L": (-1, 0),
    "R": (1, 0),
    "U": (0, -1),
    "D": (0, 1),
}

DEFAULT_BORDER_WIDTH = 1
DEFAULT_GRID_LINE_WIDTH = 1
DEFAULT_PIXELS_PER_CELL = 12
DEFAULT_VIDEO_FPS = 5.0
DEFAULT_VIDEO_CRF = 34
DEFAULT_VIDEO_PRESET = "slow"


def hex_to_rgb(hex_string: str) -> Color:
    return tuple(int(hex_string[i : i + 2], 16) for i in (0, 2, 4))


class Colors:
    BACKGROUND = hex_to_rgb("FFFFFF")
    GRID = hex_to_rgb("E7E7E7")
    AGENT = hex_to_rgb("00DC00")
    AGENT_BORDER = hex_to_rgb("006E00")
    AGENT_WALL = hex_to_rgb("FAC71E")
    AGENT_WALL_BORDER = hex_to_rgb("7D640F")
    GOAL = None
    GOAL_BORDER = hex_to_rgb("B90000")
    GOAL_OBJECT = hex_to_rgb("DC0000")
    GOAL_OBJECT_BORDER = hex_to_rgb("6E0000")
    MOVABLE = hex_to_rgb("469BFF")
    MOVABLE_BORDER = hex_to_rgb("23487F")
    WALL = hex_to_rgb("0A0A0A")
    WALL_BORDER = hex_to_rgb("050505")


@dataclass(frozen=True)
class PushWorldObject:
    position: Point
    fill_color: Color | None
    border_color: Color
    cells: frozenset[Point]


def subtract_from_points(points: Iterable[Point], offset: Point) -> frozenset[Point]:
    dx, dy = offset
    return frozenset((x - dx, y - dy) for x, y in points)


def translate_points(points: Iterable[Point], offset: Point) -> frozenset[Point]:
    dx, dy = offset
    return frozenset((x + dx, y + dy) for x, y in points)


def token_to_output_id(token_id: str) -> str:
    if token_id == "a":
        return "A"
    if token_id == "aw":
        return "AW"
    if token_id == "w":
        return "W"
    return token_id.upper()


def token_sort_key(token_text: str) -> tuple[int, str]:
    if token_text == "AW":
        return (0, token_text)
    if token_text == "W":
        return (1, token_text)
    if token_text.startswith("G"):
        return (2, token_text)
    if token_text == "A":
        return (3, token_text)
    if token_text.startswith("M"):
        return (4, token_text)
    return (5, token_text)


def write_rgb_png(rgb_bytes: bytes, width: int, height: int, png_path: Path) -> None:
    expected_size = width * height * 3
    if len(rgb_bytes) != expected_size:
        raise ValueError(
            f"expected {expected_size} RGB bytes for a {width}x{height} image, "
            f"got {len(rgb_bytes)}"
        )

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    stride = width * 3
    raw_scanlines = bytearray()
    for row_start in range(0, len(rgb_bytes), stride):
        raw_scanlines.append(0)
        raw_scanlines.extend(rgb_bytes[row_start : row_start + stride])

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", checksum)
        )

    payload = bytearray()
    payload.extend(b"\x89PNG\r\n\x1a\n")
    payload.extend(
        make_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
    )
    payload.extend(make_chunk(b"IDAT", zlib.compress(bytes(raw_scanlines), level=9)))
    payload.extend(make_chunk(b"IEND", b""))
    png_path.write_bytes(bytes(payload))


class RgbCanvas:
    def __init__(self, width: int, height: int, background: Color) -> None:
        self.width = width
        self.height = height
        row = bytes(background) * width
        self._buffer = bytearray(row * height)
        self._row_stride = width * 3

    def fill_rect(self, x1: int, y1: int, x2: int, y2: int, color: Color) -> None:
        if x1 >= x2 or y1 >= y2:
            return
        segment = bytes(color) * (x2 - x1)
        for y in range(y1, y2):
            start = y * self._row_stride + x1 * 3
            self._buffer[start : start + len(segment)] = segment

    def to_bytes(self) -> bytes:
        return bytes(self._buffer)


class PushWorldPuzzleVideo:
    def __init__(self, file_path: Path) -> None:
        obj_pixels: OrderedDict[str, set[Point]] = OrderedDict()
        elems_per_row = -1
        num_rows = 0

        with file_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                line_elems = line.split()
                if line_idx == 1:
                    elems_per_row = len(line_elems)
                elif elems_per_row != len(line_elems):
                    raise ValueError(
                        f"Row {line_idx} does not have the same number of elements as the first row."
                    )

                for x, cell_text in enumerate(line_elems, start=1):
                    for elem_id in cell_text.split("+"):
                        normalized_id = elem_id.lower()
                        if normalized_id == ".":
                            continue
                        obj_pixels.setdefault(normalized_id, set()).add((x, line_idx))
                num_rows = line_idx

        if elems_per_row <= 0 or num_rows <= 0:
            raise ValueError(f"puzzle {file_path} is empty")
        if "a" not in obj_pixels:
            raise ValueError("Every puzzle must include an agent object 'A'.")

        obj_pixels.setdefault("aw", set())
        obj_pixels.setdefault("w", set())

        self._width = elems_per_row + 2
        self._height = num_rows + 2

        for x in range(self._width):
            obj_pixels["w"].add((x, 0))
            obj_pixels["w"].add((x, self._height - 1))
        for y in range(self._height):
            obj_pixels["w"].add((0, y))
            obj_pixels["w"].add((self._width - 1, y))

        movables = ["a"]
        object_positions: dict[str, Point] = {}
        movable_objects: list[PushWorldObject] = []
        goals: list[PushWorldObject] = []
        goal_state: list[Point] = []
        goal_ids: list[str] = []
        walls: PushWorldObject | None = None
        agent_walls: PushWorldObject | None = None

        for elem_id in sorted(obj_pixels.keys(), reverse=True):
            pixels = obj_pixels[elem_id]
            if elem_id in {"w", "aw"}:
                position = (0, 0)
            else:
                xs, ys = zip(*pixels)
                position = (min(xs), min(ys))

            normalized_pixels = subtract_from_points(pixels, position)
            object_positions[elem_id] = position
            obj_pixels[elem_id] = set(normalized_pixels)

            if elem_id == "w":
                walls = PushWorldObject(
                    position=position,
                    fill_color=Colors.WALL,
                    border_color=Colors.WALL_BORDER,
                    cells=normalized_pixels,
                )
            elif elem_id == "aw":
                agent_walls = PushWorldObject(
                    position=position,
                    fill_color=Colors.AGENT_WALL,
                    border_color=Colors.AGENT_WALL_BORDER,
                    cells=normalized_pixels,
                )
            elif elem_id == "a":
                movable_objects.append(
                    PushWorldObject(
                        position=position,
                        fill_color=Colors.AGENT,
                        border_color=Colors.AGENT_BORDER,
                        cells=normalized_pixels,
                    )
                )
            elif elem_id.startswith("g"):
                goals.append(
                    PushWorldObject(
                        position=position,
                        fill_color=Colors.GOAL,
                        border_color=Colors.GOAL_BORDER,
                        cells=normalized_pixels,
                    )
                )
                goal_state.append(position)
                goal_ids.append(elem_id)
                movable_id = "m" + elem_id[1:]
                if movable_id not in obj_pixels:
                    raise ValueError(f"Goal has no associated movable object: {movable_id}")
                movables.append(movable_id)

        for elem_id in obj_pixels:
            if elem_id.startswith("m") and elem_id not in movables:
                movables.append(elem_id)

        for movable_index, elem_id in enumerate(movables[1:]):
            movable_objects.append(
                PushWorldObject(
                    position=object_positions[elem_id],
                    fill_color=(
                        Colors.MOVABLE
                        if movable_index >= len(goal_state)
                        else Colors.GOAL_OBJECT
                    ),
                    border_color=(
                        Colors.MOVABLE_BORDER
                        if movable_index >= len(goal_state)
                        else Colors.GOAL_OBJECT_BORDER
                    ),
                    cells=frozenset(obj_pixels[elem_id]),
                )
            )

        if walls is None:
            raise ValueError("puzzle has no walls")

        self._walls = walls
        self._agent_walls = agent_walls
        self._movable_objects = movable_objects
        self._goals = goals
        self._goal_state = tuple(goal_state)
        self._goal_ids = tuple(goal_ids)
        self._movable_ids = tuple(movables)
        self._source_width = elems_per_row
        self._source_height = num_rows
        self._object_positions_by_id = dict(object_positions)
        self._object_cells_by_id = {
            elem_id: frozenset(points) for elem_id, points in obj_pixels.items()
        }
        self._initial_state = tuple(object_positions[elem_id] for elem_id in movables)
        self._wall_positions = frozenset(obj_pixels["w"])
        self._agent_wall_positions = frozenset(obj_pixels["aw"])

    @property
    def initial_state(self) -> tuple[Point, ...]:
        return self._initial_state

    @property
    def movable_ids(self) -> tuple[str, ...]:
        return self._movable_ids

    @property
    def frame_width(self) -> int:
        return self._width

    @property
    def frame_height(self) -> int:
        return self._height

    def is_goal_state(self, state: tuple[Point, ...]) -> bool:
        return state[1 : 1 + len(self._goal_state)] == self._goal_state

    def get_next_state(self, state: tuple[Point, ...], action_char: str) -> tuple[Point, ...]:
        if action_char not in ACTION_DISPLACEMENTS:
            raise ValueError(f"unsupported action {action_char!r}")

        displacement = ACTION_DISPLACEMENTS[action_char]
        moved = {0}
        frontier = [0]
        occupied = [
            translate_points(obj.cells, position)
            for obj, position in zip(self._movable_objects, state)
        ]

        while frontier:
            movable_index = frontier.pop()
            shifted = translate_points(occupied[movable_index], displacement)

            if movable_index == 0:
                if shifted & (self._wall_positions | self._agent_wall_positions):
                    return state
            elif shifted & self._wall_positions:
                return state

            for obstacle_index in range(1, len(self._movable_objects)):
                if obstacle_index in moved:
                    continue
                if shifted & occupied[obstacle_index]:
                    moved.add(obstacle_index)
                    frontier.append(obstacle_index)

        dx, dy = displacement
        next_state = list(state)
        for movable_index in moved:
            x, y = state[movable_index]
            next_state[movable_index] = (x + dx, y + dy)
        return tuple(next_state)

    def apply_plan(
        self,
        plan_text: str,
        state: tuple[Point, ...] | None = None,
    ) -> tuple[Point, ...]:
        current_state = self._initial_state if state is None else state
        for action_char in normalize_plan_text(plan_text):
            current_state = self.get_next_state(current_state, action_char)
        return current_state

    def render_frame(
        self,
        state: tuple[Point, ...],
        border_width: int = DEFAULT_BORDER_WIDTH,
        grid_line_width: int = DEFAULT_GRID_LINE_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    ) -> bytes:
        if border_width < 1:
            raise ValueError("border_width must be at least 1")
        if grid_line_width < 0:
            raise ValueError("grid_line_width must be non-negative")
        if pixels_per_cell < border_width * 2 + 1:
            raise ValueError("pixels_per_cell is too small for the requested border width")

        canvas = RgbCanvas(
            width=self._width * pixels_per_cell,
            height=self._height * pixels_per_cell,
            background=Colors.BACKGROUND,
        )

        if grid_line_width:
            for x in range(0, canvas.width, pixels_per_cell):
                canvas.fill_rect(x, 0, min(x + grid_line_width, canvas.width), canvas.height, Colors.GRID)
            for y in range(0, canvas.height, pixels_per_cell):
                canvas.fill_rect(0, y, canvas.width, min(y + grid_line_width, canvas.height), Colors.GRID)

        if self._agent_walls is not None:
            self._draw_object(canvas, self._agent_walls, self._agent_walls.position, pixels_per_cell, border_width)
        self._draw_object(canvas, self._walls, self._walls.position, pixels_per_cell, border_width)

        for movable, position in zip(self._movable_objects, state):
            self._draw_object(canvas, movable, position, pixels_per_cell, border_width)
        for goal in self._goals:
            self._draw_object(canvas, goal, goal.position, pixels_per_cell, border_width)

        return canvas.to_bytes()

    def render_png(
        self,
        png_path: Path,
        state: tuple[Point, ...] | None = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        grid_line_width: int = DEFAULT_GRID_LINE_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    ) -> None:
        render_state = self._initial_state if state is None else state
        frame = self.render_frame(
            render_state,
            border_width=border_width,
            grid_line_width=grid_line_width,
            pixels_per_cell=pixels_per_cell,
        )
        write_rgb_png(
            frame,
            width=self._width * pixels_per_cell,
            height=self._height * pixels_per_cell,
            png_path=png_path,
        )

    def serialize_state(self, state: tuple[Point, ...]) -> str:
        if len(state) != len(self._movable_ids):
            raise ValueError(
                f"state length {len(state)} does not match puzzle movable count "
                f"{len(self._movable_ids)}"
            )

        cells: list[list[list[str]]] = [
            [[] for _ in range(self._source_width)] for _ in range(self._source_height)
        ]

        def add_object(token_id: str, position: Point) -> None:
            rendered_token = token_to_output_id(token_id)
            for cell_x, cell_y in self._object_cells_by_id[token_id]:
                x = position[0] + cell_x
                y = position[1] + cell_y
                if not (1 <= x <= self._source_width and 1 <= y <= self._source_height):
                    if token_id == "w":
                        continue
                    raise ValueError(
                        f"object {rendered_token} extends outside the puzzle interior "
                        f"at {(x, y)}"
                    )
                cells[y - 1][x - 1].append(rendered_token)

        if "aw" in self._object_cells_by_id:
            add_object("aw", self._object_positions_by_id["aw"])
        if "w" in self._object_cells_by_id:
            add_object("w", self._object_positions_by_id["w"])
        for goal_id in self._goal_ids:
            add_object(goal_id, self._object_positions_by_id[goal_id])
        for movable_id, position in zip(self._movable_ids, state):
            add_object(movable_id, position)

        rows: list[str] = []
        for row in cells:
            tokens = [
                "." if not cell_parts else "+".join(sorted(cell_parts, key=token_sort_key))
                for cell_parts in row
            ]
            rows.append(" ".join(tokens))
        return "\n".join(rows) + "\n"

    def _draw_object(
        self,
        canvas: RgbCanvas,
        obj: PushWorldObject,
        position: Point,
        pixels_per_cell: int,
        border_width: int,
    ) -> None:
        border_offsets = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        for cell_x, cell_y in obj.cells:
            pixel_x = (position[0] + cell_x) * pixels_per_cell
            pixel_y = (position[1] + cell_y) * pixels_per_cell

            if obj.fill_color is not None:
                canvas.fill_rect(
                    pixel_x,
                    pixel_y,
                    pixel_x + pixels_per_cell,
                    pixel_y + pixels_per_cell,
                    obj.fill_color,
                )

            for delta_row, delta_col in border_offsets:
                neighbor = (cell_x + delta_col, cell_y + delta_row)
                if neighbor in obj.cells:
                    continue
                row_start = pixel_y + max(0, delta_row) * (pixels_per_cell - border_width)
                row_end = row_start + (pixels_per_cell if delta_row == 0 else border_width)
                col_start = pixel_x + max(0, delta_col) * (pixels_per_cell - border_width)
                col_end = col_start + (pixels_per_cell if delta_col == 0 else border_width)
                canvas.fill_rect(col_start, row_start, col_end, row_end, obj.border_color)


def normalize_plan_text(plan_text: str) -> str:
    normalized = "".join(ch for ch in plan_text.upper() if not ch.isspace())
    invalid = sorted({ch for ch in normalized if ch not in ACTION_DISPLACEMENTS})
    if invalid:
        invalid_text = ", ".join(repr(ch) for ch in invalid)
        raise ValueError(f"plan contains invalid action characters: {invalid_text}")
    return normalized


def render_solution_video(
    puzzle_path: Path,
    solution_text: str,
    video_path: Path,
    fps: float = DEFAULT_VIDEO_FPS,
    pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    border_width: int = DEFAULT_BORDER_WIDTH,
    grid_line_width: int = DEFAULT_GRID_LINE_WIDTH,
    crf: int = DEFAULT_VIDEO_CRF,
    preset: str = DEFAULT_VIDEO_PRESET,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if crf < 0:
        raise ValueError("crf must be non-negative")

    puzzle = PushWorldPuzzleVideo(Path(puzzle_path))
    plan = normalize_plan_text(solution_text)
    frame_width = puzzle._width * pixels_per_cell
    frame_height = puzzle._height * pixels_per_cell

    if frame_width % 2 != 0 or frame_height % 2 != 0:
        raise ValueError(
            f"video dimensions must be even, got {frame_width}x{frame_height}; "
            f"choose an even pixels_per_cell value"
        )

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{frame_width}x{frame_height}",
        "-framerate",
        f"{fps:.3f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("ffmpeg did not expose a writable stdin pipe")

        state = puzzle.initial_state
        process.stdin.write(
            puzzle.render_frame(
                state,
                border_width=border_width,
                grid_line_width=grid_line_width,
                pixels_per_cell=pixels_per_cell,
            )
        )
        frames_written = 1

        for action_char in plan:
            if puzzle.is_goal_state(state):
                raise ValueError("plan reaches the goal before the final action")
            state = puzzle.get_next_state(state, action_char)
            process.stdin.write(
                puzzle.render_frame(
                    state,
                    border_width=border_width,
                    grid_line_width=grid_line_width,
                    pixels_per_cell=pixels_per_cell,
                )
            )
            frames_written += 1

        if not puzzle.is_goal_state(state):
            raise ValueError("plan does not solve the puzzle")

        if frames_written == 1:
            process.stdin.write(
                puzzle.render_frame(
                    state,
                    border_width=border_width,
                    grid_line_width=grid_line_width,
                    pixels_per_cell=pixels_per_cell,
                )
            )

        process.stdin.close()
        stderr_output = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            detail = stderr_output.strip() or f"ffmpeg exited with code {return_code}"
            raise RuntimeError(f"failed to render {video_path}: {detail}")
    except FileNotFoundError as error:
        if error.filename == "ffmpeg":
            raise RuntimeError(
                "Rendering solution videos requires `ffmpeg` to be installed."
            ) from error
        raise
    except Exception:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait()
        if video_path.exists():
            video_path.unlink()
        raise


def render_solution_video_from_files(
    puzzle_path: Path,
    solution_path: Path,
    video_path: Path,
    fps: float = DEFAULT_VIDEO_FPS,
    pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    border_width: int = DEFAULT_BORDER_WIDTH,
    grid_line_width: int = DEFAULT_GRID_LINE_WIDTH,
    crf: int = DEFAULT_VIDEO_CRF,
    preset: str = DEFAULT_VIDEO_PRESET,
) -> None:
    solution_text = Path(solution_path).read_text(encoding="utf-8")
    render_solution_video(
        puzzle_path=Path(puzzle_path),
        solution_text=solution_text,
        video_path=Path(video_path),
        fps=fps,
        pixels_per_cell=pixels_per_cell,
        border_width=border_width,
        grid_line_width=grid_line_width,
        crf=crf,
        preset=preset,
    )


def render_puzzle_png(
    puzzle_path: Path,
    png_path: Path,
    plan_text: str = "",
    pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    border_width: int = DEFAULT_BORDER_WIDTH,
    grid_line_width: int = DEFAULT_GRID_LINE_WIDTH,
) -> None:
    puzzle = PushWorldPuzzleVideo(Path(puzzle_path))
    state = puzzle.apply_plan(plan_text)
    puzzle.render_png(
        png_path=Path(png_path),
        state=state,
        pixels_per_cell=pixels_per_cell,
        border_width=border_width,
        grid_line_width=grid_line_width,
    )
