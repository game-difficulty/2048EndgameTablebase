from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

from .common import bytes_sha256, write_json, write_jsonl


def _node_text(node: ET.Element) -> str:
    return "".join(node.itertext()).replace("\u00a0", " ").strip()


def _number(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _join_line_fragments(fragments: list[dict[str, Any]]) -> str:
    fragments = sorted(fragments, key=lambda item: (item["left"], item["top"]))
    result = ""
    previous_right: int | None = None
    for fragment in fragments:
        text = str(fragment["text"])
        if not text:
            continue
        if result and previous_right is not None:
            gap = fragment["left"] - previous_right
            ascii_boundary = result[-1:].isascii() and text[:1].isascii()
            if gap >= 5 or (gap >= 2 and ascii_boundary):
                result += " "
        result += text
        previous_right = max(
            previous_right or 0,
            fragment["left"] + fragment["width"],
        )
    return result.strip()


def _group_text_lines(
    page: ET.Element,
    fonts: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    fonts = fonts or {
        node.attrib.get("id", ""): {
            "size": _number(node.attrib.get("size")),
            "color": str(node.attrib.get("color") or "#000000").lower(),
        }
        for node in page.findall("fontspec")
    }
    fragments: list[dict[str, Any]] = []
    for node in page.findall("text"):
        text = _node_text(node)
        if not text:
            continue
        font = fonts.get(node.attrib.get("font", ""), {})
        fragments.append(
            {
                "text": text,
                "top": _number(node.attrib.get("top")),
                "left": _number(node.attrib.get("left")),
                "width": _number(node.attrib.get("width")),
                "height": _number(node.attrib.get("height")),
                "size": int(font.get("size") or 0),
                "color": str(font.get("color") or "#000000"),
            }
        )

    groups: list[list[dict[str, Any]]] = []
    for fragment in sorted(fragments, key=lambda item: (item["top"], item["left"])):
        matching = next(
            (
                group
                for group in reversed(groups[-3:])
                if abs(min(item["top"] for item in group) - fragment["top"]) <= 2
            ),
            None,
        )
        if matching is None:
            groups.append([fragment])
        else:
            matching.append(fragment)

    lines: list[dict[str, Any]] = []
    for group in groups:
        left = min(item["left"] for item in group)
        top = min(item["top"] for item in group)
        right = max(item["left"] + item["width"] for item in group)
        bottom = max(item["top"] + item["height"] for item in group)
        dominant = max(group, key=lambda item: (item["size"], len(item["text"])))
        lines.append(
            {
                "text": _join_line_fragments(group),
                "top": top,
                "left": left,
                "width": right - left,
                "height": bottom - top,
                "size": max(item["size"] for item in group),
                "color": dominant["color"],
            }
        )
    return sorted(lines, key=lambda item: (item["top"], item["left"]))


def _resolve_image_path(xml_path: Path, source: str) -> Path:
    source_path = Path(source)
    candidates = [
        source_path,
        xml_path.parent / source_path,
        xml_path.parent / source_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"PDF image asset not found: {source}")


def extract_pdf_media_from_xml(
    pdf_path: Path,
    xml_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    if media_dir.exists():
        resolved_media = media_dir.resolve()
        if resolved_media.parent != out_dir.resolve():
            raise RuntimeError(f"Refusing to replace unexpected media path: {resolved_media}")
        shutil.rmtree(resolved_media)
    media_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(xml_path).getroot()
    global_fonts = {
        node.attrib.get("id", ""): {
            "size": _number(node.attrib.get("size")),
            "color": str(node.attrib.get("color") or "#000000").lower(),
        }
        for page in root.findall("page")
        for node in page.findall("fontspec")
    }
    pages: list[dict[str, Any]] = []
    pending_images: list[dict[str, Any]] = []
    for page in root.findall("page"):
        page_number = _number(page.attrib.get("number"))
        page_payload = {
            "number": page_number,
            "width": _number(page.attrib.get("width")),
            "height": _number(page.attrib.get("height")),
            "lines": _group_text_lines(page, global_fonts),
            "images": [],
        }
        for node in page.findall("image"):
            pending_images.append(
                {
                    "page": page_number,
                    "top": _number(node.attrib.get("top")),
                    "left": _number(node.attrib.get("left")),
                    "display_width": _number(node.attrib.get("width")),
                    "display_height": _number(node.attrib.get("height")),
                    "source_path": _resolve_image_path(xml_path, str(node.attrib.get("src") or "")),
                    "page_payload": page_payload,
                }
            )
        pages.append(page_payload)

    pending_images.sort(key=lambda item: (item["page"], item["top"], item["left"]))
    all_lines = [
        {
            **line,
            "page": page["number"],
            "sort_key": page["number"] * 100000 + line["top"],
        }
        for page in pages
        for line in page["lines"]
    ]
    all_lines.sort(key=lambda line: line["sort_key"])
    line_keys = [line["sort_key"] for line in all_lines]

    records: list[dict[str, Any]] = []
    first_seen_by_sha: dict[str, str] = {}
    for doc_index, item in enumerate(pending_images):
        source_path = item["source_path"]
        data = source_path.read_bytes()
        sha256 = bytes_sha256(data)
        suffix = source_path.suffix.lower() or ".png"
        image_id = f"img_{doc_index:04d}"
        output_path = media_dir / f"{image_id}{suffix}"
        output_path.write_bytes(data)
        with Image.open(output_path) as image:
            width, height = int(image.width), int(image.height)

        sort_key = item["page"] * 100000 + item["top"]
        insertion = bisect.bisect_left(line_keys, sort_key)
        before = [line["text"] for line in all_lines[max(0, insertion - 3):insertion]]
        after = [line["text"] for line in all_lines[insertion:insertion + 3]]
        duplicate_of = first_seen_by_sha.get(sha256)
        if duplicate_of is None:
            first_seen_by_sha[sha256] = image_id

        placement = {
            "image_id": image_id,
            "top": item["top"],
            "left": item["left"],
            "width": item["display_width"],
            "height": item["display_height"],
        }
        item["page_payload"]["images"].append(placement)
        records.append(
            {
                "image_id": image_id,
                "doc_index": doc_index,
                "source_name": source_path.name,
                "sha256": sha256,
                "duplicate_of": duplicate_of,
                "path": str(output_path),
                "relative_path": output_path.relative_to(out_dir).as_posix(),
                "width": width,
                "height": height,
                "page": item["page"],
                "top": item["top"],
                "left": item["left"],
                "display_width": item["display_width"],
                "display_height": item["display_height"],
                "paragraph_index": doc_index,
                "context_before": before,
                "context_after": after,
            }
        )

    write_jsonl(out_dir / "media_manifest.jsonl", records)
    write_json(
        out_dir / "media_summary.json",
        {
            "pdf": str(pdf_path),
            "pages": len(pages),
            "image_occurrences": len(records),
            "unique_images": len(first_seen_by_sha),
            "duplicates": sum(1 for record in records if record["duplicate_of"]),
        },
    )
    write_json(
        out_dir / "pdf_content.json",
        {
            "schema_version": 1,
            "source_name": pdf_path.name,
            "pages": pages,
        },
    )
    return records


def extract_pdf_media(pdf_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    executable = shutil.which("pdftohtml")
    if not executable:
        raise RuntimeError("pdftohtml (Poppler) is required for PDF guide extraction.")
    scratch_dir = out_dir / ".pdf_extract"
    if scratch_dir.exists():
        resolved_scratch = scratch_dir.resolve()
        if resolved_scratch.parent != out_dir.resolve():
            raise RuntimeError(f"Refusing to replace unexpected scratch path: {resolved_scratch}")
        shutil.rmtree(resolved_scratch)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    xml_path = scratch_dir / "source.xml"
    subprocess.run(
        [
            executable,
            "-q",
            "-xml",
            "-hidden",
            "-nodrm",
            "-zoom",
            "1.5",
            str(pdf_path.resolve()),
            str(xml_path.resolve()),
        ],
        check=True,
    )
    return extract_pdf_media_from_xml(pdf_path, xml_path, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract guide images and text layout from a PDF file.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    records = extract_pdf_media(args.pdf, args.out)
    summary = json.loads((args.out / "media_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
