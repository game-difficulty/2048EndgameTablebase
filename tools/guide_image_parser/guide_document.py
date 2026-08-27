from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json
from .docx_media import NS, _paragraph_text


CHAPTER_RE = re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*章(?:\s+|$)")
SECTION_RE = re.compile(r"^第\s*\d+\s*节")
CAPTION_RE = re.compile(r"^图\s*\d+(?:\s*[-－]\s*\d+)?(?:\s|$)")


def _normalized_guide_hex(value: Any, visible_rows: int, visible_cols: int = 4) -> str:
    text = str(value or "").strip().lower().removeprefix("0x")
    text = "".join(char for char in text if char in "0123456789abcdef")
    visible_digits = max(0, min(16, int(visible_rows) * int(visible_cols)))
    if len(text) == visible_digits and visible_digits < 16:
        text = text + "f" * (16 - visible_digits)
    if len(text) < 16:
        text = text.ljust(16, "f")
    return text[:16]


def _asset_path(entry: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(str(entry.get("path") or ""))
    if path.exists():
        return path
    relative_path = entry.get("relative_path")
    if relative_path:
        fallback = manifest_path.parent / str(relative_path)
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Guide image not found for {entry.get('image_id')}: {path}")


def _board_payload(
    board: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    visible_rows = int(board.get("visible_rows") or 4)
    visible_cols = int(board.get("visible_cols") or 4)
    raw_bbox = [int(value) for value in (board.get("bbox") or [0, 0, 0, 0])]
    raw_bbox = (raw_bbox + [0, 0, 0, 0])[:4]
    x = max(0, min(raw_bbox[0], image_width))
    y = max(0, min(raw_bbox[1], image_height))
    width = max(0, min(raw_bbox[2], image_width - x))
    height = max(0, min(raw_bbox[3], image_height - y))
    return {
        "board_id": str(board.get("board_id") or ""),
        "bbox": [x, y, width, height],
        "visible_rows": visible_rows,
        "visible_cols": visible_cols,
        "hex": _normalized_guide_hex(board.get("hex"), visible_rows, visible_cols),
        "confidence": round(float(board.get("confidence") or 0.0), 4),
        "flags": list(board.get("flags") or []),
    }


def _update_index(
    index_path: Path,
    *,
    document_id: str,
    title: str,
    language: str,
    document_path: Path,
) -> None:
    payload: dict[str, Any] = {"version": 1, "documents": []}
    if index_path.exists():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded)
    documents = [
        entry
        for entry in payload.get("documents", [])
        if isinstance(entry, dict) and entry.get("id") != document_id
    ]
    source = document_path.relative_to(index_path.parent).as_posix()
    documents.append(
        {
            "id": document_id,
            "title": title,
            "language": language,
            "type": "guide",
            "source": source,
        }
    )
    payload["version"] = 1
    payload["documents"] = documents
    write_json(index_path, payload)


def build_guide_document(
    *,
    docx_path: Path,
    manifest_path: Path,
    parsed_path: Path,
    out_dir: Path,
    document_id: str,
    title: str,
    language: str = "zh",
    index_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    parsed_by_id = {
        record["image_id"]: record
        for record in read_jsonl(parsed_path)
    }
    images_by_paragraph: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        images_by_paragraph[int(entry.get("paragraph_index") or 0)].append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        paragraphs = document.findall(".//w:p", NS)
        body = document.find(".//w:body", NS)
        body_paragraph_ids = {
            id(paragraph)
            for paragraph in (body.findall("w:p", NS) if body is not None else [])
        }

    first_chapter_index = next(
        (
            index
            for index, paragraph in enumerate(paragraphs)
            if CHAPTER_RE.match(_paragraph_text(paragraph))
        ),
        0,
    )

    blocks: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    chapter_number = 0
    section_number = 0
    copied_images: set[str] = set()
    board_count = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph_index < first_chapter_index:
            continue
        if id(paragraph) not in body_paragraph_ids:
            continue

        text = _paragraph_text(paragraph)
        paragraph_images = images_by_paragraph.get(paragraph_index, [])

        if text:
            if CHAPTER_RE.match(text):
                chapter_number += 1
                section_number = 0
                anchor_id = f"guide-chapter-{chapter_number:02d}"
                block = {
                    "type": "heading",
                    "level": 1,
                    "id": anchor_id,
                    "text": text,
                    "paragraph_index": paragraph_index,
                }
                blocks.append(block)
                toc.append({"level": 1, "id": anchor_id, "text": text})
            elif SECTION_RE.match(text):
                section_number += 1
                anchor_id = f"guide-chapter-{chapter_number:02d}-section-{section_number:02d}"
                block = {
                    "type": "heading",
                    "level": 2,
                    "id": anchor_id,
                    "text": text,
                    "paragraph_index": paragraph_index,
                }
                blocks.append(block)
                toc.append({"level": 2, "id": anchor_id, "text": text})
            elif CAPTION_RE.match(text) and blocks and blocks[-1].get("type") == "figure":
                blocks[-1]["caption"] = text
            else:
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": text,
                        "paragraph_index": paragraph_index,
                    }
                )

        for image_entry in paragraph_images:
            image_id = str(image_entry["image_id"])
            source_path = _asset_path(image_entry, manifest_path)
            suffix = source_path.suffix.lower() or ".bin"
            asset_name = f"{image_id}{suffix}"
            if asset_name not in copied_images:
                shutil.copyfile(source_path, media_dir / asset_name)
                copied_images.add(asset_name)

            parsed = parsed_by_id.get(image_id, {})
            image_width = int(image_entry.get("width") or parsed.get("width") or 0)
            image_height = int(image_entry.get("height") or parsed.get("height") or 0)
            boards = [
                _board_payload(board, image_width, image_height)
                for board in parsed.get("boards", [])
            ]
            board_count += len(boards)
            blocks.append(
                {
                    "type": "figure",
                    "image_id": image_id,
                    "src": f"media/{asset_name}",
                    "width": image_width,
                    "height": image_height,
                    "boards": boards,
                    "paragraph_index": paragraph_index,
                }
            )

    payload = {
        "schema_version": 1,
        "id": document_id,
        "type": "guide",
        "title": title,
        "language": language,
        "source_name": docx_path.name,
        "toc": toc,
        "blocks": blocks,
        "stats": {
            "chapters": chapter_number,
            "sections": sum(1 for item in toc if item["level"] == 2),
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
    parser = argparse.ArgumentParser(description="Build a browser guide document from DOCX parser output.")
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parsed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()

    result = build_guide_document(
        docx_path=args.docx,
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
