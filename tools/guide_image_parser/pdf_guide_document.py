from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json
from .guide_document import _board_payload, _update_index


CHAPTER_RE = re.compile(r"^[一二三四五六七八九十百]+、")
SECTION_RE = re.compile(r"^\d+、")
SUBSECTION_RE = re.compile(r"^[（(]\d+[）)]")
ROMAN_PAGE_RE = re.compile(r"^(?:[ivxlcdm]+|\d+)$", re.IGNORECASE)


def _heading_level(line: dict[str, Any]) -> int | None:
    text = str(line.get("text") or "").strip()
    color = str(line.get("color") or "").lower()
    size = int(line.get("size") or 0)
    is_teal = color in {"#0e4660", "#0e465f"}
    if text in {"定义", "前言"} and is_teal and size >= 30:
        return 1
    if CHAPTER_RE.match(text) and is_teal and size >= 33:
        return 1
    if SECTION_RE.match(text) and is_teal and size >= 28:
        return 2
    if SUBSECTION_RE.match(text) and is_teal and size >= 23:
        return 3
    if is_teal and size >= 23:
        return 3
    return None


def _join_wrapped_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1:].isascii() and right[:1].isascii() and left[-1:].isalnum() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _included_pages(pages: list[dict[str, Any]]) -> set[int]:
    toc_page = next(
        (
            page["number"]
            for page in pages
            if any(str(line.get("text") or "").strip() == "目录" for line in page.get("lines", []))
        ),
        None,
    )
    first_chapter_page = next(
        (
            page["number"]
            for page in pages
            if toc_page is None or int(page["number"]) > int(toc_page)
            if any(_heading_level(line) == 1 for line in page.get("lines", []))
        ),
        None,
    )
    included: set[int] = set()
    for page in pages:
        number = int(page["number"])
        if number == 1:
            continue
        if toc_page is not None and first_chapter_page is not None and toc_page <= number < first_chapter_page:
            continue
        included.add(number)
    return included


def _is_footer(line: dict[str, Any], page_height: int) -> bool:
    text = str(line.get("text") or "").strip()
    return bool(
        int(line.get("top") or 0) >= page_height * 0.88
        and ROMAN_PAGE_RE.fullmatch(text)
    )


def build_pdf_guide_document(
    *,
    pdf_path: Path,
    content_path: Path,
    manifest_path: Path,
    parsed_path: Path,
    out_dir: Path,
    document_id: str,
    title: str,
    language: str = "zh",
    index_path: Path | None = None,
) -> dict[str, Any]:
    content = json.loads(content_path.read_text(encoding="utf-8"))
    pages = list(content.get("pages") or [])
    included_pages = _included_pages(pages)
    manifest = read_jsonl(manifest_path)
    manifest_by_id = {entry["image_id"]: entry for entry in manifest}
    parsed_by_id = {record["image_id"]: record for record in read_jsonl(parsed_path)}

    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    blocks: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    copied_images: set[str] = set()
    board_count = 0
    heading_count = 0
    sequence_index = 0

    def append_paragraph(parts: list[str], page_number: int) -> None:
        nonlocal sequence_index
        text = "".join(parts).strip()
        if not text:
            return
        blocks.append(
            {
                "type": "paragraph",
                "text": text,
                "page": page_number,
                "paragraph_index": sequence_index,
            }
        )
        sequence_index += 1

    for page in pages:
        page_number = int(page.get("number") or 0)
        if page_number not in included_pages:
            continue
        page_height = int(page.get("height") or 0)
        events = [
            {"type": "line", **line}
            for line in page.get("lines", [])
            if not _is_footer(line, page_height)
        ]
        events.extend({"type": "image", **image} for image in page.get("images", []))
        events.sort(key=lambda event: (int(event.get("top") or 0), 0 if event["type"] == "line" else 1))

        paragraph_parts: list[str] = []
        previous_line: dict[str, Any] | None = None
        previous_color = ""
        for event in events:
            if event["type"] == "image":
                append_paragraph(paragraph_parts, page_number)
                paragraph_parts = []
                previous_line = None

                image_id = str(event.get("image_id") or "")
                entry = manifest_by_id.get(image_id)
                if not entry:
                    continue
                source_path = Path(str(entry.get("path") or ""))
                if not source_path.exists() and entry.get("relative_path"):
                    source_path = manifest_path.parent / str(entry["relative_path"])
                if not source_path.exists():
                    raise FileNotFoundError(f"Guide image not found: {image_id}")
                suffix = source_path.suffix.lower() or ".png"
                asset_name = f"{image_id}{suffix}"
                if asset_name not in copied_images:
                    shutil.copyfile(source_path, media_dir / asset_name)
                    copied_images.add(asset_name)
                parsed = parsed_by_id.get(image_id, {})
                width = int(entry.get("width") or parsed.get("width") or 0)
                height = int(entry.get("height") or parsed.get("height") or 0)
                boards = [
                    _board_payload(board, width, height)
                    for board in parsed.get("boards", [])
                ]
                board_count += len(boards)
                blocks.append(
                    {
                        "type": "figure",
                        "image_id": image_id,
                        "src": f"media/{asset_name}",
                        "width": width,
                        "height": height,
                        "boards": boards,
                        "page": page_number,
                        "paragraph_index": sequence_index,
                    }
                )
                sequence_index += 1
                continue

            text = str(event.get("text") or "").strip()
            if not text or text == "目录":
                continue
            level = _heading_level(event)
            if level is not None:
                append_paragraph(paragraph_parts, page_number)
                paragraph_parts = []
                heading_count += 1
                anchor_id = f"{document_id}-heading-{heading_count:03d}"
                blocks.append(
                    {
                        "type": "heading",
                        "level": level,
                        "id": anchor_id,
                        "text": text,
                        "page": page_number,
                        "paragraph_index": sequence_index,
                    }
                )
                toc.append({"level": level, "id": anchor_id, "text": text})
                sequence_index += 1
                previous_line = None
                previous_color = ""
                continue

            color = str(event.get("color") or "").lower()
            top_gap = 0
            if previous_line is not None:
                top_gap = int(event.get("top") or 0) - (
                    int(previous_line.get("top") or 0) + int(previous_line.get("height") or 0)
                )
            starts_indented = int(event.get("left") or 0) > int(page.get("width") or 0) * 0.18
            color_break = bool(previous_color and color != previous_color and color != "#000000")
            paragraph_break = bool(
                paragraph_parts
                and (
                    top_gap > 34
                    or color_break
                    or (starts_indented and paragraph_parts[-1].endswith(("。", "！", "？")))
                )
            )
            if paragraph_break:
                append_paragraph(paragraph_parts, page_number)
                paragraph_parts = []
            if paragraph_parts:
                paragraph_parts[-1] = _join_wrapped_text(paragraph_parts[-1], text)
            else:
                paragraph_parts.append(text)
            previous_line = event
            previous_color = color

        append_paragraph(paragraph_parts, page_number)

    payload = {
        "schema_version": 1,
        "id": document_id,
        "type": "guide",
        "title": title,
        "language": language,
        "source_name": pdf_path.name,
        "toc": toc,
        "blocks": blocks,
        "stats": {
            "pages": len(included_pages),
            "headings": len(toc),
            "images": len(copied_images),
            "boards": board_count,
        },
    }
    document_path = out_dir / "document.json"
    write_json(document_path, payload)
    if index_path is not None:
        _update_index(
            index_path,
            document_id=document_id,
            title=title,
            language=language,
            document_path=document_path,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a browser guide document from PDF parser output.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parsed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    result = build_pdf_guide_document(
        pdf_path=args.pdf,
        content_path=args.content,
        manifest_path=args.manifest,
        parsed_path=args.parsed,
        out_dir=args.out,
        document_id=args.document_id,
        title=args.title,
        language=args.language,
        index_path=args.index,
    )
    print(json.dumps(result["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
