#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "master_summary.json"
DEFAULT_SVG = REPO_ROOT / "master_summary_solved_curve.svg"
DEFAULT_PNG = REPO_ROOT / "master_summary_solved_curve.png"

SWIFT_SVG_TO_PNG = r"""
import AppKit
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 5 else {
    fputs("usage: render_svg.swift <input-svg> <output-png> <width> <height>\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])
guard let width = Double(arguments[3]), let height = Double(arguments[4]) else {
    fputs("width and height must be numeric\n", stderr)
    exit(2)
}

guard let image = NSImage(contentsOf: inputURL) else {
    fputs("failed to load SVG from \(inputURL.path)\n", stderr)
    exit(1)
}

let targetSize = NSSize(width: width, height: height)
guard let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: Int(width),
    pixelsHigh: Int(height),
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bitmapFormat: [],
    bytesPerRow: 0,
    bitsPerPixel: 0
) else {
    fputs("failed to create bitmap buffer\n", stderr)
    exit(1)
}

bitmap.size = targetSize
NSGraphicsContext.saveGraphicsState()
guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
    fputs("failed to create graphics context\n", stderr)
    exit(1)
}

NSGraphicsContext.current = context
NSColor.white.setFill()
NSBezierPath(rect: NSRect(origin: .zero, size: targetSize)).fill()
image.draw(in: NSRect(origin: .zero, size: targetSize))
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()

guard let data = bitmap.representation(using: .png, properties: [:]) else {
    fputs("failed to encode PNG data\n", stderr)
    exit(1)
}

do {
    try data.write(to: outputURL)
} catch {
    fputs("failed to write PNG to \(outputURL.path): \(error)\n", stderr)
    exit(1)
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cumulative solved-vs-time plot from master_summary.json "
            "and render both SVG and PNG artifacts."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--svg-output", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--x-min", type=float, default=1e-4)
    parser.add_argument("--x-max", type=float, default=1e3)
    parser.add_argument("--total-puzzles", type=int, default=223)
    parser.add_argument("--max-percent", type=int, default=80)
    parser.add_argument("--legend-label", type=str, default="Recursive Heuristic")
    parser.add_argument(
        "--y-minimum",
        type=int,
        default=170,
        help="Requested lower bound for the chart ceiling; the actual ceiling grows if the data needs it.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} did not contain a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def solved_durations(entries: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for entry in entries:
        if not entry.get("solved"):
            continue
        duration = entry.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            durations.append(float(duration))
    durations.sort()
    return durations


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 20,
    anchor: str = "middle",
    fill: str = "#222222",
    weight: str = "normal",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>'
        f"{escape(text)}</text>"
    )


def write_plot_svg(
    *,
    durations: list[float],
    svg_path: Path,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    total_puzzles: int,
    max_percent: int,
    legend_label: str,
    y_minimum: int,
) -> tuple[int, int]:
    total = len(durations)
    if total == 0:
        raise ValueError("No solved puzzles with positive duration were found")

    percent_ceiling_count = total_puzzles * (max_percent / 100.0)
    y_max = max(
        y_minimum,
        total,
        int(math.ceil(percent_ceiling_count / 5.0) * 5),
    )
    margin_left = 120
    margin_right = 120
    margin_top = 40
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_left = margin_left
    plot_top = margin_top
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    log_min = math.log10(x_min)
    log_max = math.log10(x_max)

    def x_pos(value: float) -> float:
        clamped = min(max(value, x_min), x_max)
        ratio = (math.log10(clamped) - log_min) / (log_max - log_min)
        return plot_left + ratio * plot_width

    def y_pos(value: float) -> float:
        ratio = value / y_max
        return plot_bottom - ratio * plot_height

    x_ticks = [10**exp for exp in range(int(log_min), int(log_max) + 1)]
    y_ticks = list(range(0, y_max + 1, 20))
    if y_ticks[-1] != y_max:
        y_ticks.append(y_max)

    right_ticks = list(range(0, max_percent + 1, 10))
    if right_ticks[-1] != max_percent:
        right_ticks.append(max_percent)

    polyline_points: list[str] = []
    first_x = x_pos(durations[0])
    polyline_points.append(f"{first_x:.2f},{y_pos(0):.2f}")
    for index, duration in enumerate(durations, start=1):
        polyline_points.append(f"{x_pos(duration):.2f},{y_pos(index):.2f}")

    marker_indices = {
        index
        for index in range(1, total + 1)
        if index % 20 == 0 or index in {1, total}
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" '
            'stroke="#d4d4d4" stroke-width="1.5" stroke-dasharray="4 6" />'
        )
        parts.append(svg_text(plot_left - 18, y + 7, str(tick), size=22, anchor="end"))

    for tick in x_ticks:
        x = x_pos(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_top:.2f}" x2="{x:.2f}" y2="{plot_bottom:.2f}" '
            'stroke="#d4d4d4" stroke-width="1.5" stroke-dasharray="4 6" />'
        )
        exp = int(round(math.log10(tick)))
        parts.append(svg_text(x, plot_bottom + 46, f"10^{exp}", size=22))

    parts.extend(
        [
            f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_bottom:.2f}" stroke="#222222" stroke-width="3" />',
            f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#222222" stroke-width="3" />',
            f'<line x1="{plot_right:.2f}" y1="{plot_top:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#222222" stroke-width="3" />',
        ]
    )

    for percent in right_ticks:
        count_value = total_puzzles * percent / 100.0
        if count_value > y_max:
            continue
        parts.append(svg_text(plot_right + 18, y_pos(count_value) + 7, str(percent), size=22, anchor="start"))

    parts.append(
        f'<polyline points="{" ".join(polyline_points)}" fill="none" '
        'stroke="#d62728" stroke-width="5" stroke-linejoin="round" stroke-linecap="round" />'
    )

    for index, duration in enumerate(durations, start=1):
        if index not in marker_indices:
            continue
        x = x_pos(duration)
        y = y_pos(index)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7.5" fill="#d62728" stroke="#ffffff" stroke-width="2.5" />')

    legend_x = plot_left + 18
    legend_y = plot_top + 20
    legend_w = 270
    legend_h = 72
    parts.extend(
        [
            f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_w}" height="{legend_h}" rx="6" ry="6" fill="#fafafa" stroke="#c9c9c9" stroke-width="2" opacity="0.96" />',
            f'<line x1="{legend_x + 18:.2f}" y1="{legend_y + 36:.2f}" x2="{legend_x + 76:.2f}" y2="{legend_y + 36:.2f}" stroke="#d62728" stroke-width="5" stroke-linecap="round" />',
            f'<circle cx="{legend_x + 47:.2f}" cy="{legend_y + 36:.2f}" r="7.5" fill="#d62728" stroke="#ffffff" stroke-width="2.5" />',
            svg_text(legend_x + 92, legend_y + 44, legend_label, size=24, anchor="start"),
        ]
    )

    parts.extend(
        [
            svg_text((plot_left + plot_right) / 2, height - 28, "Planning Time Per Puzzle (seconds)", size=28),
            svg_text(42, (plot_top + plot_bottom) / 2, "Number of Puzzles Solved", size=28, rotate=-90),
            svg_text(width - 32, (plot_top + plot_bottom) / 2, "% of Puzzles Solved", size=28, rotate=90),
        ]
    )

    parts.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return total, y_max


def render_png_from_svg(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svg-render-") as temp_dir:
        swift_path = Path(temp_dir) / "render_svg.swift"
        swift_path.write_text(SWIFT_SVG_TO_PNG, encoding="utf-8")
        subprocess.run(
            [
                "swift",
                "-module-cache-path",
                "/tmp/swift-module-cache",
                str(swift_path),
                str(svg_path.resolve()),
                str(png_path.resolve()),
                str(width),
                str(height),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def main() -> int:
    args = parse_args()
    entries = load_summary(args.input)
    durations = solved_durations(entries)
    total, y_max = write_plot_svg(
        durations=durations,
        svg_path=args.svg_output,
        width=args.width,
        height=args.height,
        x_min=args.x_min,
        x_max=args.x_max,
        total_puzzles=args.total_puzzles,
        max_percent=args.max_percent,
        legend_label=args.legend_label,
        y_minimum=args.y_minimum,
    )
    render_png_from_svg(args.svg_output, args.png_output, args.width, args.height)
    print(
        f"Rendered {total} solved puzzles from {args.input} to "
        f"{args.svg_output} and {args.png_output} with y_max={y_max}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
