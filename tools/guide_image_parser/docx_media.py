from __future__ import annotations

import argparse
import posixpath
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .common import IMAGE_EXTENSIONS, bytes_sha256, write_json, write_jsonl


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


@dataclass(frozen=True)
class ParagraphImage:
    paragraph_index: int
    relationship_id: str


def _relationship_targets(archive: zipfile.ZipFile) -> dict[str, str]:
    rels_path = "word/_rels/document.xml.rels"
    root = ET.fromstring(archive.read(rels_path))
    targets: dict[str, str] = {}
    for rel in root.findall("rel:Relationship", NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if not rel_id or not target:
            continue
        normalized = posixpath.normpath(posixpath.join("word", target))
        targets[rel_id] = normalized
    return targets


def _paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()


def _paragraph_images(paragraph: ET.Element, paragraph_index: int) -> list[ParagraphImage]:
    images: list[ParagraphImage] = []
    for blip in paragraph.findall(".//a:blip", NS):
        rel_id = blip.attrib.get(f"{{{NS['r']}}}embed") or blip.attrib.get(
            f"{{{NS['r']}}}link"
        )
        if rel_id:
            images.append(ParagraphImage(paragraph_index, rel_id))
    return images


def _image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _context_for(paragraphs: list[str], paragraph_index: int, window: int = 3) -> dict[str, list[str]]:
    before: list[str] = []
    after: list[str] = []

    for index in range(paragraph_index - 1, -1, -1):
        if paragraphs[index]:
            before.append(paragraphs[index])
        if len(before) >= window:
            break

    for index in range(paragraph_index + 1, len(paragraphs)):
        if paragraphs[index]:
            after.append(paragraphs[index])
        if len(after) >= window:
            break

    return {"before": list(reversed(before)), "after": after}


def extract_docx_media(docx_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    first_seen_by_sha: dict[str, str] = {}

    with zipfile.ZipFile(docx_path) as archive:
        relationship_targets = _relationship_targets(archive)
        document = ET.fromstring(archive.read("word/document.xml"))
        paragraphs = document.findall(".//w:p", NS)
        paragraph_texts = [_paragraph_text(paragraph) for paragraph in paragraphs]

        occurrences: list[ParagraphImage] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            occurrences.extend(_paragraph_images(paragraph, paragraph_index))

        for doc_index, occurrence in enumerate(occurrences):
            source_name = relationship_targets.get(occurrence.relationship_id, "")
            suffix = Path(source_name).suffix.lower() or ".bin"
            if not source_name or suffix not in IMAGE_EXTENSIONS:
                continue
            data = archive.read(source_name)
            sha256 = bytes_sha256(data)
            image_id = f"img_{len(records):04d}"
            output_name = f"{image_id}{suffix}"
            output_path = media_dir / output_name
            output_path.write_bytes(data)
            width, height = _image_dimensions(output_path)
            duplicate_of = first_seen_by_sha.get(sha256)
            if duplicate_of is None:
                first_seen_by_sha[sha256] = image_id

            context = _context_for(paragraph_texts, occurrence.paragraph_index)
            records.append(
                {
                    "image_id": image_id,
                    "doc_index": doc_index,
                    "source_name": source_name,
                    "relationship_id": occurrence.relationship_id,
                    "sha256": sha256,
                    "duplicate_of": duplicate_of,
                    "path": str(output_path),
                    "relative_path": str(output_path.relative_to(out_dir)).replace("\\", "/"),
                    "width": width,
                    "height": height,
                    "paragraph_index": occurrence.paragraph_index,
                    "context_before": context["before"],
                    "context_after": context["after"],
                }
            )

    write_jsonl(out_dir / "media_manifest.jsonl", records)
    write_json(
        out_dir / "media_summary.json",
        {
            "docx": str(docx_path),
            "image_occurrences": len(records),
            "unique_images": len(first_seen_by_sha),
            "duplicates": sum(1 for record in records if record["duplicate_of"]),
        },
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract guide images from a DOCX file.")
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = extract_docx_media(args.docx, args.out)
    print(f"Wrote {len(records)} image records to {args.out / 'media_manifest.jsonl'}")


if __name__ == "__main__":
    main()

