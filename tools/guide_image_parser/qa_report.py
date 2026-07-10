from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from .common import data_uri, read_jsonl


TILE_COLORS = {
    0: "#f6dfe4",
    1: "#f2d8dd",
    2: "#efd4d9",
    3: "#efb3bb",
    4: "#f79aa1",
    5: "#ef7d85",
    6: "#e65e66",
    7: "#f7b44b",
    8: "#ff9500",
    9: "#f8c847",
    10: "#d8b042",
    11: "#b48d38",
    12: "#99742f",
    13: "#775928",
    14: "#5f4623",
    15: "#ebeef2",
}

CELL_SIZE = 38
GRID_GAP = 4


def _style() -> str:
    return """
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }
h1 { margin: 0 0 16px; }
.summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.pill { border: 1px solid #cbd5e1; border-radius: 999px; padding: 4px 10px; font-weight: 700; background: #f8fafc; }
details { border: 1px solid #d8dee9; border-radius: 8px; margin: 12px 0; background: white; }
details[open] { box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }
summary { cursor: pointer; padding: 12px 14px; font-weight: 800; }
.record { display: grid; grid-template-columns: minmax(280px, 48%) minmax(320px, 1fr); gap: 16px; padding: 0 14px 14px; }
.image-wrap { position: relative; display: inline-block; max-width: 100%; border: 1px solid #e2e8f0; background: #fff; }
.image-wrap img { display: block; max-width: 100%; height: auto; }
.overlay { position: absolute; inset: 0; pointer-events: none; }
.board-box { position: absolute; border: 2px solid #2563eb; box-sizing: border-box; }
.anno-box { position: absolute; border: 2px solid #dc2626; box-sizing: border-box; }
.board-card { border: 1px solid #d8dee9; border-radius: 8px; padding: 10px; margin-bottom: 10px; background: #f8fafc; }
.board-grid-wrap { position: relative; margin-top: 8px; }
.board-grid { display: grid; grid-auto-rows: 38px; gap: 4px; }
.cell { display: flex; align-items: center; justify-content: center; border-radius: 4px; font-weight: 800; font-size: 12px; color: #5b4a42; }
.cell.pad { color: #64748b; background: #edf2f7 !important; }
.board-overlay { position: absolute; inset: 0; pointer-events: none; overflow: visible; }
.annotation-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.annotation-badge { border: 1px solid #cbd5e1; border-radius: 999px; padding: 2px 7px; background: #fff; font-size: 11px; color: #334155; }
.flags { color: #b45309; font-size: 12px; margin-top: 6px; }
.json { white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 10px; border-radius: 8px; font-size: 12px; max-height: 220px; overflow: auto; }
.status-accepted summary { color: #166534; }
.status-review summary { color: #9a3412; }
.status-ignored summary { color: #64748b; }
@media (max-width: 900px) { .record { grid-template-columns: 1fr; } }
"""


def _bbox_style(bbox: list[int], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in bbox]
    return (
        f"left:{x / width * 100:.4f}%;top:{y / height * 100:.4f}%;"
        f"width:{w / width * 100:.4f}%;height:{h / height * 100:.4f}%;"
    )


def _cell_label(cell: dict[str, Any]) -> str:
    if "padding_f" in (cell.get("flags") or []):
        return "f"
    value = int(cell.get("value") or 0)
    if value <= 0:
        return ""
    if value >= 1024:
        return f"{value // 1024}k" if value % 1024 == 0 else str(value)
    return str(value)


def _grid_extent(count: int) -> int:
    count = max(1, int(count))
    return count * CELL_SIZE + max(0, count - 1) * GRID_GAP


def _cell_center_px(
    position: list[int] | tuple[int, int],
) -> tuple[float, float]:
    row, col = int(position[0]), int(position[1])
    return (
        col * (CELL_SIZE + GRID_GAP) + CELL_SIZE / 2,
        row * (CELL_SIZE + GRID_GAP) + CELL_SIZE / 2,
    )


def _highlight_rect_px(
    cells: list[list[int]],
) -> tuple[float, float, float, float] | None:
    if not cells:
        return None
    row_indices = [int(cell[0]) for cell in cells]
    col_indices = [int(cell[1]) for cell in cells]
    row0, row1 = min(row_indices), max(row_indices)
    col0, col1 = min(col_indices), max(col_indices)
    if len(cells) != (row1 - row0 + 1) * (col1 - col0 + 1):
        return None
    return (
        col0 * (CELL_SIZE + GRID_GAP),
        row0 * (CELL_SIZE + GRID_GAP),
        (col1 - col0 + 1) * CELL_SIZE + (col1 - col0) * GRID_GAP,
        (row1 - row0 + 1) * CELL_SIZE + (row1 - row0) * GRID_GAP,
    )


def _render_board_overlay(board: dict[str, Any], annotations: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    board_id = board.get("board_id")
    rows = max(1, int(board.get("visible_rows") or 4))
    cols = max(1, int(board.get("visible_cols") or 4))
    grid_width = _grid_extent(cols)
    grid_height = _grid_extent(rows)
    for annotation in annotations:
        if annotation.get("type") != "highlight" or annotation.get("board_id") != board_id:
            continue
        visible_cells = [
            cell
            for cell in (annotation.get("cells") or [])
            if int(cell[0]) < rows and int(cell[1]) < cols
        ]
        rect = _highlight_rect_px(visible_cells)
        if rect is None:
            continue
        x, y, width, height = rect
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            'fill="none" stroke="#dc2626" stroke-width="2.6" vector-effect="non-scaling-stroke" />'
        )
    for annotation in annotations:
        if annotation.get("type") != "arrow" or annotation.get("board_id") != board_id:
            continue
        from_cell = annotation.get("from_cell")
        to_cell = annotation.get("to_cell")
        if not from_cell or not to_cell:
            continue
        if int(from_cell[0]) >= rows or int(to_cell[0]) >= rows:
            continue
        if int(from_cell[1]) >= cols or int(to_cell[1]) >= cols:
            continue
        x1, y1 = _cell_center_px(from_cell)
        x2, y2 = _cell_center_px(to_cell)
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#7c3aed" stroke-width="3" stroke-linecap="round" marker-end="url(#arrowhead)" />'
        )
    if not lines:
        return ""
    return f"""
<svg class="board-overlay" viewBox="0 0 {grid_width} {grid_height}" preserveAspectRatio="none" aria-hidden="true">
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="#7c3aed"></path>
    </marker>
  </defs>
  {''.join(lines)}
</svg>
"""


def _annotation_badge(annotation: dict[str, Any]) -> str:
    annotation_type = html.escape(str(annotation.get("type", "")))
    confidence = html.escape(str(annotation.get("confidence", "")))
    if annotation.get("type") == "connector":
        label = f'{annotation_type}: {html.escape(str(annotation.get("from_board")))} -> {html.escape(str(annotation.get("to_board")))}'
    elif annotation.get("type") == "highlight":
        label = f'{annotation_type}: {len(annotation.get("cells") or [])} cells'
    elif annotation.get("type") == "arrow":
        label = f'{annotation_type}: {html.escape(str(annotation.get("from_cell")))} -> {html.escape(str(annotation.get("to_cell")))}'
    else:
        label = annotation_type
    return f'<span class="annotation-badge" title="confidence {confidence}">{label}</span>'


def _render_board(board: dict[str, Any], annotations: list[dict[str, Any]]) -> str:
    rows = max(1, int(board.get("visible_rows") or 4))
    cols = max(1, int(board.get("visible_cols") or 4))
    cells = sorted(
        [
            cell
            for cell in (board.get("cells") or [])
            if int(cell.get("row", 0)) < rows and int(cell.get("col", 0)) < cols
        ],
        key=lambda item: (item["row"], item["col"]),
    )
    board_id = board.get("board_id")
    board_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("board_id") == board_id
        or annotation.get("from_board") == board_id
        or annotation.get("to_board") == board_id
    ]
    items = []
    for cell in cells:
        exponent = int(cell.get("exponent") or 0)
        classes = "cell"
        if "padding_f" in (cell.get("flags") or []):
            classes += " pad"
        color = TILE_COLORS.get(exponent, "#e2e8f0")
        label = html.escape(_cell_label(cell))
        title = html.escape(
            f"r{cell.get('row')} c{cell.get('col')} exp={exponent} conf={cell.get('confidence')}"
        )
        items.append(f'<div class="{classes}" title="{title}" style="background:{color}">{label}</div>')
    flags = board.get("flags") or []
    flag_html = f'<div class="flags">flags: {html.escape(", ".join(flags))}</div>' if flags else ""
    overlay_html = _render_board_overlay(board, board_annotations)
    grid_width = _grid_extent(cols)
    grid_height = _grid_extent(rows)
    wrap_style = f"width:{grid_width}px;height:{grid_height}px;"
    grid_style = f"grid-template-columns:repeat({cols}, {CELL_SIZE}px);grid-auto-rows:{CELL_SIZE}px;gap:{GRID_GAP}px;"
    badges_html = (
        f'<div class="annotation-badges">{"".join(_annotation_badge(annotation) for annotation in board_annotations)}</div>'
        if board_annotations
        else ""
    )
    return f"""
<div class="board-card">
  <div><strong>{html.escape(board.get("board_id", ""))}</strong> hex <code>{html.escape(board.get("hex", ""))}</code></div>
  <div>visible {board.get("visible_rows")}x{board.get("visible_cols")} · confidence {board.get("confidence")}</div>
  <div class="board-grid-wrap" style="{wrap_style}"><div class="board-grid" style="{grid_style}">{''.join(items)}</div>{overlay_html}</div>
  {badges_html}
  {flag_html}
</div>
"""


def _render_record(record: dict[str, Any]) -> str:
    image_path = Path(record["path"])
    image_src = data_uri(image_path) if image_path.exists() else ""
    width = int(record.get("width") or 1)
    height = int(record.get("height") or 1)
    board_boxes = [
        f'<div class="board-box" title="{html.escape(board.get("board_id", ""))}" style="{_bbox_style(board["bbox"], width, height)}"></div>'
        for board in record.get("boards", [])
        if board.get("bbox")
    ]
    annotation_boxes = [
        f'<div class="anno-box" title="{html.escape(annotation.get("type", ""))}" style="{_bbox_style(annotation["bbox"], width, height)}"></div>'
        for annotation in record.get("annotations", [])
        if annotation.get("bbox")
    ]
    boards_html = "".join(_render_board(board, record.get("annotations", [])) for board in record.get("boards", []))
    quality = html.escape(str(record.get("quality", {})))
    annotations = html.escape(str(record.get("annotations", [])))
    open_attr = " open" if record.get("status") == "review" else ""
    status = html.escape(record.get("status", ""))
    title = (
        f'{html.escape(record.get("image_id", ""))} · {status} · '
        f'{len(record.get("boards", []))} boards · {len(record.get("annotations", []))} annotations'
    )
    return f"""
<details class="status-{status}"{open_attr}>
  <summary>{title}</summary>
  <div class="record">
    <div>
      <div class="image-wrap">
        <img src="{image_src}" width="{width}" height="{height}" alt="{html.escape(record.get("image_id", ""))}" />
        <div class="overlay">{''.join(board_boxes)}{''.join(annotation_boxes)}</div>
      </div>
    </div>
    <div>
      {boards_html or '<p>No boards detected.</p>'}
      <h3>Quality</h3>
      <div class="json">{quality}</div>
      <h3>Annotations</h3>
      <div class="json">{annotations}</div>
    </div>
  </div>
</details>
"""


def render_qa_report(parsed_path: Path, out_path: Path) -> None:
    records = read_jsonl(parsed_path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.get("status", "unknown")] = counts.get(record.get("status", "unknown"), 0) + 1
    summary = "".join(
        f'<span class="pill">{html.escape(key)}: {count}</span>'
        for key, count in sorted(counts.items())
    )
    body = "\n".join(_render_record(record) for record in records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Guide Image Parser QA</title>
  <style>{_style()}</style>
</head>
<body>
  <h1>Guide Image Parser QA</h1>
  <div class="summary">{summary}</div>
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render guide image parser QA report.")
    parser.add_argument("--parsed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render_qa_report(args.parsed, args.out)
    print(f"Wrote QA report to {args.out}")


if __name__ == "__main__":
    main()
