from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.guide_image_parser.common import read_jsonl
from tools.guide_image_parser.docx_media import extract_docx_media
from tools.guide_image_parser.guide_document import build_guide_document
from tools.guide_image_parser.image_parser import (
    BoardCandidate,
    CellProbe,
    DEFAULT_GUIDE_PALETTE,
    TextMatch,
    _cell_record,
    _dedupe_line_annotations,
    _mask_lattice_board_candidates,
    _repair_edge_shifted_cell_grids,
    parse_manifest,
)
from tools.guide_image_parser.qa_report import render_qa_report
from tools.guide_image_parser.validate_gold_samples import validate_gold_samples


REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_PATH = REPO_ROOT / "font" / "ClearSans" / "ClearSans-Bold.ttf"
COLORS = {
    0: "#f8dde2",
    1: "#f3d7db",
    2: "#f0d2d6",
    3: "#efb5bd",
    4: "#f79ba3",
    5: "#ed7f87",
    6: "#df666b",
    7: "#efb064",
    8: "#f89523",
    9: "#efc954",
    10: "#d8b042",
    11: "#b78c34",
    12: "#98712e",
    13: "#795826",
    14: "#60451f",
    15: "#eceff3",
}


def _font(size: int):
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    x0, y0, x1, y1 = box
    font_size = 26
    while font_size > 7:
        font = _font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= (x1 - x0) * 0.82 and bbox[3] - bbox[1] <= (y1 - y0) * 0.7:
            break
        font_size -= 1
    font = _font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - width) / 2 - bbox[0]
    y = y0 + (y1 - y0 - height) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill="#7f6270")


def draw_board(exponents: list[list[int]], cell: int = 38, gap: int = 4, pad: int = 3) -> Image.Image:
    rows = len(exponents)
    width = pad * 2 + 4 * cell + 3 * gap
    height = pad * 2 + rows * cell + (rows - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        for col in range(4):
            exponent = exponents[row][col]
            x0 = pad + col * (cell + gap)
            y0 = pad + row * (cell + gap)
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=COLORS[exponent])
            if exponent > 0:
                _draw_centered_text(draw, (x0, y0, x1, y1), str(2**exponent))
    return image


def make_manifest(tmp: Path, images: list[Image.Image]) -> Path:
    records = []
    media_dir = tmp / "media"
    media_dir.mkdir()
    for index, image in enumerate(images):
        image_id = f"img_{index:04d}"
        path = media_dir / f"{image_id}.png"
        image.save(path)
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "image_id": image_id,
                "doc_index": index,
                "source_name": f"word/media/image{index + 1}.png",
                "sha256": sha,
                "duplicate_of": None,
                "path": str(path),
                "relative_path": f"media/{image_id}.png",
                "width": image.width,
                "height": image.height,
                "paragraph_index": index,
                "context_before": [],
                "context_after": [],
            }
        )
    manifest = tmp / "media_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


class GuideImageParserTests(unittest.TestCase):
    def test_build_guide_document_preserves_toc_images_and_partial_hex(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            docx_path = tmp / "guide.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Front matter</w:t></w:r></w:p>
    <w:p><w:r><w:t>第一章 测试章节</w:t></w:r></w:p>
    <w:p><w:r><w:t>第 1 节 测试小节</w:t></w:r></w:p>
    <w:p />
    <w:p><w:r><w:t>图 1-1</w:t></w:r></w:p>
    <w:p><w:r><w:t>正文内容</w:t></w:r></w:p>
    <w:sectPr />
  </w:body>
</w:document>
"""
            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            image_path = tmp / "img_0000.png"
            image_path.write_bytes(b"original-image-bytes")
            manifest_path = tmp / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_id": "img_0000",
                        "path": str(image_path),
                        "relative_path": image_path.name,
                        "paragraph_index": 3,
                        "width": 100,
                        "height": 80,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parsed_path = tmp / "parsed.jsonl"
            parsed_path.write_text(
                json.dumps(
                    {
                        "image_id": "img_0000",
                        "boards": [
                            {
                                "board_id": "img_0000_b00",
                                "bbox": [10, 10, 95, 75],
                                "visible_rows": 3,
                                "visible_cols": 4,
                                "hex": "001064127ff5",
                                "confidence": 0.9,
                                "flags": [],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_dir = tmp / "public" / "guides" / "test-guide"
            index_path = out_dir.parent / "index.json"

            result = build_guide_document(
                docx_path=docx_path,
                manifest_path=manifest_path,
                parsed_path=parsed_path,
                out_dir=out_dir,
                document_id="test-guide",
                title="Test Guide",
                index_path=index_path,
            )

            self.assertEqual(result["stats"], {"chapters": 1, "sections": 1, "images": 1, "boards": 1})
            self.assertEqual([item["level"] for item in result["toc"]], [1, 2])
            figure = next(block for block in result["blocks"] if block["type"] == "figure")
            self.assertEqual(figure["caption"], "图 1-1")
            self.assertEqual(figure["boards"][0]["bbox"], [10, 10, 90, 70])
            self.assertEqual(figure["boards"][0]["hex"], "001064127ff5ffff")
            self.assertEqual((out_dir / figure["src"]).read_bytes(), b"original-image-bytes")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["documents"][0]["source"], "test-guide/document.json")

    def test_gold_validation_supports_targeted_board_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            parsed_path = tmp / "parsed.jsonl"
            parsed_path.write_text(
                json.dumps(
                    {
                        "image_id": "img_test",
                        "boards": [
                            {
                                "board_id": "img_test_b00",
                                "bbox": [10, 10, 120, 60],
                                "visible_rows": 2,
                                "visible_cols": 4,
                                "hex": "12345678ffffffff",
                            },
                            {
                                "board_id": "img_test_b01",
                                "bbox": [160, 10, 120, 60],
                                "visible_rows": 2,
                                "visible_cols": 4,
                                "hex": "87654321ffffffff",
                            },
                        ],
                        "annotations": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gold_path = tmp / "gold.yaml"
            gold_path.write_text(
                """samples:
  - image_id: img_test
    board_count: 2
    boards:
      - board_id: img_test_b01
        bbox: [161, 10, 120, 60]
        bbox_tolerance: 1
        visible_rows: 2
        hex: \"87654321ffffffff\"
""",
                encoding="utf-8",
            )

            result = validate_gold_samples(parsed_path, gold_path)

            self.assertEqual(result["failures"], [])
            self.assertEqual(result["boards"], {"passed": 1, "total": 1})
            self.assertEqual(result["board_counts"], {"passed": 1, "total": 1})

    def test_full_board_hex(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            exponents = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [0, 1, 2, 3],
            ]
            manifest = make_manifest(tmp, [draw_board(exponents)])
            parsed_path = tmp / "parsed_images.jsonl"
            parsed = parse_manifest(manifest, parsed_path)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["boards"][0]["hex"], "123456789abc0123")

    def test_guide_palette_values_do_not_drop_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            exponents = [
                [1, 2, 2, 1],
                [2, 3, 2, 3],
                [8, 6, 5, 7],
                [12, 11, 9, 10],
            ]
            manifest = make_manifest(tmp, [draw_board(exponents, cell=64, gap=9, pad=8)])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(parsed[0]["boards"][0]["hex"], "122123238657cb9a")

    def test_red_frame_on_empty_cell_does_not_create_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[0, 1, 0, 0], [0, 0, 0, 0]], cell=44)
            draw = ImageDraw.Draw(image)
            draw.rectangle((2, 2, 49, 49), outline="#d42626", width=2)
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(parsed[0]["boards"][0]["hex"][:4], "0100")

    def test_two_occluded_by_horizontal_arrow(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[0, 0, 1, 0], [0, 1, 2, 4]], cell=44)
            draw = ImageDraw.Draw(image)
            y = 3 + 22
            draw.line((3, y, image.width - 4, y), fill="#9d74c8", width=2)
            draw.polygon([(3, y), (11, y - 5), (11, y + 5)], fill="#9d74c8")
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(parsed[0]["boards"][0]["hex"][:8], "00100124")

    def test_twos_are_reconstructed_after_straight_arrow_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            cell = 30
            gap = 4
            pad = 3
            image = draw_board([[1, 0, 0, 1], [5, 4, 3, 2]], cell=cell, gap=gap, pad=pad)
            draw = ImageDraw.Draw(image)
            y = pad + cell // 2
            draw.line((pad, y, image.width - pad, y), fill="#9d74c8", width=4)
            draw.polygon(
                [(image.width - pad, y), (image.width - pad - 8, y - 6), (image.width - pad - 8, y + 6)],
                fill="#9d74c8",
            )
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")

            self.assertEqual(parsed[0]["boards"][0]["hex"][:4], "1001")

    def test_reconstructed_two_cannot_override_confident_four_color(self) -> None:
        cell = CellProbe(
            row=0,
            col=0,
            bbox=[0, 0, 30, 30],
            background_rgb=list(DEFAULT_GUIDE_PALETTE[2]),
            text_match=TextMatch(
                exponent=1,
                value=2,
                score=0.59,
                runner_up_score=0.525,
                area_ratio=0.04,
            ),
            text_area_ratio=0.04,
        )

        record, _needs_review = _cell_record(cell, DEFAULT_GUIDE_PALETTE)

        self.assertEqual(record["exponent"], 2)

    def test_partial_board_bottom_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            exponents = [
                [1, 1, 2, 1],
                [0, 2, 3, 4],
                [9, 8, 5, 4],
            ]
            manifest = make_manifest(tmp, [draw_board(exponents)])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(parsed[0]["boards"][0]["visible_rows"], 3)
            self.assertEqual(parsed[0]["boards"][0]["hex"], "112102349854ffff")

    def test_multi_board_and_connector(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            left = draw_board([[1, 0, 0, 0], [2, 0, 0, 0]], cell=30)
            right = draw_board([[0, 0, 0, 1], [0, 0, 0, 2]], cell=30)
            canvas = Image.new("RGB", (left.width + right.width + 60, max(left.height, right.height)), "white")
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width + 60, 0))
            draw = ImageDraw.Draw(canvas)
            y = canvas.height // 2
            draw.line((left.width + 8, y, left.width + 50, y), fill="#9d74c8", width=2)
            draw.polygon([(left.width + 50, y), (left.width + 42, y - 5), (left.width + 42, y + 5)], fill="#9d74c8")
            manifest = make_manifest(tmp, [canvas])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertGreaterEqual(len(parsed[0]["boards"]), 2)
            self.assertTrue(any(item["type"] == "connector" for item in parsed[0]["annotations"]))

    def test_highlight_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[1, 2, 3, 4], [0, 0, 0, 0]], cell=34)
            draw = ImageDraw.Draw(image)
            # Highlight the second visible cell.
            x0 = 3 + 1 * (34 + 4) - 1
            y0 = 3 - 1
            draw.rectangle((x0, y0, x0 + 36, y0 + 36), outline="#d42626", width=2)
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertTrue(any(item["type"] == "highlight" for item in parsed[0]["annotations"]))

    def test_red_frame_range_maps_to_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board(
                [[1, 2, 3, 4], [2, 3, 4, 5], [0, 0, 0, 0], [0, 0, 0, 0]],
                cell=34,
            )
            draw = ImageDraw.Draw(image)
            x0 = 3 - 1
            y0 = 3 - 1
            x1 = 3 + 4 * 34 + 3 * 4 + 1
            y1 = 3 + 2 * 34 + 1 * 4 + 1
            draw.rectangle((x0, y0, x1, y1), outline="#d42626", width=2)
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            expected = [[row, col] for row in (0, 1) for col in range(4)]
            highlights = [item for item in parsed[0]["annotations"] if item["type"] == "highlight"]
            self.assertEqual([item.get("cells") for item in highlights], [expected])

    def test_internal_red_separators_split_into_row_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board(
                [[1, 0, 0, 0], [2, 1, 1, 0], [3, 4, 5, 6], [10, 9, 8, 7]],
                cell=34,
            )
            draw = ImageDraw.Draw(image)
            cell = 34
            gap = 4
            pad = 3
            x0 = pad - 1
            x1 = pad + 4 * cell + 3 * gap + 1
            for row in range(4):
                y0 = pad + row * (cell + gap) - 1
                y1 = y0 + cell + 2
                draw.rectangle((x0, y0, x1, y1), outline="#d42626", width=2)
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            highlights = [item for item in parsed[0]["annotations"] if item["type"] == "highlight"]
            expected = [[[row, col] for col in range(4)] for row in range(4)]
            self.assertEqual([item.get("cells") for item in highlights], expected)

    def test_continuous_arrow_fragments_merge(self) -> None:
        annotations = [
            {
                "type": "arrow",
                "annotation_id": "a0",
                "board_id": "b0",
                "from_cell": [0, 0],
                "to_cell": [0, 1],
                "bbox": [0, 10, 20, 2],
                "confidence": 0.7,
                "flags": ["direction_inferred_from_geometry"],
            },
            {
                "type": "arrow",
                "annotation_id": "a1",
                "board_id": "b0",
                "from_cell": [0, 1],
                "to_cell": [0, 2],
                "bbox": [20, 10, 20, 2],
                "confidence": 0.8,
                "flags": ["direction_inferred_from_geometry"],
            },
            {
                "type": "arrow",
                "annotation_id": "a2",
                "board_id": "b0",
                "from_cell": [0, 2],
                "to_cell": [0, 3],
                "bbox": [40, 10, 20, 2],
                "confidence": 0.9,
                "flags": [],
            },
        ]
        merged = [item for item in _dedupe_line_annotations(annotations) if item["type"] == "arrow"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["from_cell"], [0, 0])
        self.assertEqual(merged[0]["to_cell"], [0, 3])

    def test_contained_reverse_arrow_fragment_is_suppressed(self) -> None:
        annotations = [
            {
                "type": "arrow",
                "annotation_id": "long",
                "board_id": "b0",
                "from_cell": [0, 0],
                "to_cell": [0, 3],
                "bbox": [0, 10, 60, 2],
                "confidence": 0.95,
                "flags": [],
            },
            {
                "type": "arrow",
                "annotation_id": "short",
                "board_id": "b0",
                "from_cell": [0, 2],
                "to_cell": [0, 1],
                "bbox": [20, 10, 20, 2],
                "confidence": 0.55,
                "flags": ["direction_inferred_from_geometry"],
            },
        ]
        merged = [item for item in _dedupe_line_annotations(annotations) if item["type"] == "arrow"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["from_cell"], [0, 0])
        self.assertEqual(merged[0]["to_cell"], [0, 3])

    def test_opposite_arrow_fragments_merge_by_weighted_direction(self) -> None:
        annotations = [
            {
                "type": "arrow",
                "annotation_id": "left",
                "board_id": "b0",
                "from_cell": [3, 1],
                "to_cell": [3, 0],
                "bbox": [10, 130, 40, 4],
                "confidence": 0.62,
                "flags": [],
            },
            {
                "type": "arrow",
                "annotation_id": "mid",
                "board_id": "b0",
                "from_cell": [3, 1],
                "to_cell": [3, 2],
                "bbox": [54, 130, 40, 4],
                "confidence": 0.62,
                "flags": ["direction_inferred_from_geometry"],
            },
            {
                "type": "arrow",
                "annotation_id": "right",
                "board_id": "b0",
                "from_cell": [3, 2],
                "to_cell": [3, 3],
                "bbox": [98, 130, 40, 4],
                "confidence": 0.50,
                "flags": [],
            },
        ]
        merged = [item for item in _dedupe_line_annotations(annotations) if item["type"] == "arrow"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["from_cell"], [3, 0])
        self.assertEqual(merged[0]["to_cell"], [3, 3])

    def test_parallel_arrow_group_harmonizes_direction(self) -> None:
        annotations = [
            {
                "type": "arrow",
                "annotation_id": "short_wrong",
                "board_id": "b0",
                "from_cell": [1, 2],
                "to_cell": [0, 2],
                "bbox": [84, 18, 7, 48],
                "confidence": 0.88,
                "flags": [],
            },
            {
                "type": "arrow",
                "annotation_id": "long_right",
                "board_id": "b0",
                "from_cell": [0, 3],
                "to_cell": [1, 3],
                "bbox": [122, 4, 7, 64],
                "confidence": 0.9,
                "flags": [],
            },
        ]
        arrows = [item for item in _dedupe_line_annotations(annotations) if item["type"] == "arrow"]
        self.assertEqual(len(arrows), 2)
        self.assertTrue(all(item.get("from_cell", [None])[0] == 0 for item in arrows))
        self.assertTrue(all(item.get("to_cell", [None])[0] == 1 for item in arrows))
        self.assertTrue(any("direction_harmonized_with_parallel_group" in item.get("flags", []) for item in arrows))

    def test_blank_purple_cell_is_32768(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[0, 0, 0, 0] for _ in range(4)], cell=38)
            draw = ImageDraw.Draw(image)
            cell = 38
            gap = 4
            pad = 3
            row = 1
            col = 1
            x0 = pad + col * (cell + gap)
            y0 = pad + row * (cell + gap)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill="#f0acd0")
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(parsed[0]["boards"][0]["hex"][5], "f")

    def test_sparse_two_by_two_multi_board_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            boards = [
                draw_board([[0, 1, 0, 2], [0, 0, 1, 3]], cell=28),
                draw_board([[1, 0, 0, 2], [0, 1, 2, 3]], cell=28),
                draw_board([[0, 0, 1, 2], [1, 1, 2, 3]], cell=28),
                draw_board([[1, 2, 0, 0], [0, 1, 2, 3]], cell=28),
            ]
            canvas = Image.new("RGB", (boards[0].width * 2 + 54, boards[0].height * 2 + 34), "white")
            positions = [(0, 0), (boards[0].width + 54, 0), (0, boards[0].height + 34), (boards[0].width + 54, boards[0].height + 34)]
            for board, position in zip(boards, positions):
                canvas.paste(board, position)
            manifest = make_manifest(tmp, [canvas])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertEqual(len(parsed[0]["boards"]), 4)
            self.assertTrue(all(board["visible_rows"] == 2 for board in parsed[0]["boards"]))

    def test_color_lattice_recovers_arrow_occluded_board(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            exponents = [[2, 1, 2, 1], [5, 4, 2, 2], [9, 8, 4, 6]]
            boards = [draw_board(exponents, cell=30, gap=0, pad=0) for _ in range(5)]
            board_width, board_height = boards[0].size
            x_gap = 28
            y_gap = 18
            positions = [
                (0, 0),
                (board_width + x_gap, 0),
                (2 * (board_width + x_gap), 0),
                (0, board_height + y_gap),
                (board_width + x_gap, board_height + y_gap),
            ]
            canvas = Image.new(
                "RGB",
                (3 * board_width + 2 * x_gap, 2 * board_height + y_gap),
                "white",
            )
            for board, position in zip(boards, positions):
                canvas.paste(board, position)

            draw = ImageDraw.Draw(canvas)
            lower_left_y = board_height + y_gap
            for row in (0, 1):
                y = lower_left_y + row * 30 + 15
                draw.line((0, y, board_width - 1, y), fill="#9d74c8", width=17)
                draw.polygon([(board_width - 1, y), (board_width - 14, y - 12), (board_width - 14, y + 12)], fill="#9d74c8")

            manifest = make_manifest(tmp, [canvas])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            detected = parsed[0]["boards"]

            self.assertEqual(len(detected), 5)
            lower_left = min(
                (board for board in detected if board["bbox"][1] > board_height),
                key=lambda board: board["bbox"][0],
            )
            self.assertLess(lower_left["bbox"][0], board_width // 2)
            self.assertEqual(lower_left["visible_rows"], 3)

    def test_color_lattice_fills_missing_connected_component_candidate(self) -> None:
        cell = 30
        gap = 4
        pad = 3
        exponents = [[2, 1, 2, 1], [5, 4, 2, 2], [9, 8, 4, 6]]
        board_images = [draw_board(exponents, cell=cell, gap=gap, pad=pad) for _ in range(5)]
        board_width, board_height = board_images[0].size
        x_gap = 28
        y_gap = 18
        positions = [
            (0, 0),
            (board_width + x_gap, 0),
            (2 * (board_width + x_gap), 0),
            (0, board_height + y_gap),
            (board_width + x_gap, board_height + y_gap),
        ]
        canvas = Image.new(
            "RGB",
            (3 * board_width + 2 * x_gap, 2 * board_height + y_gap),
            "white",
        )
        for board, position in zip(board_images, positions):
            canvas.paste(board, position)

        def seed_at(x: int, y: int) -> BoardCandidate:
            cell_boxes = [
                (
                    row,
                    col,
                    [x + pad + col * (cell + gap), y + pad + row * (cell + gap), cell, cell],
                )
                for row in range(3)
                for col in range(4)
            ]
            return BoardCandidate(
                bbox=[x + pad, y + pad, 4 * cell + 3 * gap, 3 * cell + 2 * gap],
                visible_rows=3,
                visible_cols=4,
                fit_confidence=0.95,
                cell_boxes=cell_boxes,
                flags=["partial_board_bottom_f_padding"],
                source="cell_grid",
            )

        seeds = [seed_at(*positions[index]) for index in (0, 1, 2, 4)]
        inferred = _mask_lattice_board_candidates(np.asarray(canvas), seeds)

        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].bbox, seed_at(*positions[3]).bbox)
        self.assertIn("mask_lattice_inferred", inferred[0].flags)

        occupied = seeds + [
            BoardCandidate(
                **{
                    **seed_at(*positions[3]).__dict__,
                    "source": "rough",
                }
            )
        ]
        self.assertEqual(
            _mask_lattice_board_candidates(np.asarray(canvas), seeds, occupied),
            [],
        )

    def test_palette_support_repairs_one_column_board_shift(self) -> None:
        cell = 30
        gap = 4
        pad = 3
        board = draw_board([[1, 2, 3, 4], [5, 6, 7, 8]], cell=cell, gap=gap, pad=pad)
        canvas = Image.new("RGB", (board.width + cell + gap + 4, board.height), "white")
        canvas.paste(board, (0, 0))
        correct_boxes = [
            (
                row,
                col,
                [pad + col * (cell + gap), pad + row * (cell + gap), cell, cell],
            )
            for row in range(2)
            for col in range(4)
        ]
        shifted = BoardCandidate(
            bbox=[pad + cell + gap, pad, 4 * cell + 3 * gap, 2 * cell + gap],
            visible_rows=2,
            visible_cols=4,
            fit_confidence=0.76,
            cell_boxes=[
                (row, col, [bbox[0] + cell + gap, bbox[1], bbox[2], bbox[3]])
                for row, col, bbox in correct_boxes
            ],
            flags=["partial_board_bottom_f_padding"],
            source="cell_grid",
        )

        repaired = _repair_edge_shifted_cell_grids(np.asarray(canvas), [shifted])

        self.assertEqual(repaired[0].bbox[0], pad)
        self.assertEqual(repaired[0].cell_boxes, correct_boxes)
        self.assertIn("edge_column_shift_recovered", repaired[0].flags)

    def test_connector_and_internal_arrow_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            left = draw_board([[1, 0, 0, 0], [2, 0, 0, 0]], cell=30)
            right = draw_board([[0, 0, 0, 1], [0, 0, 0, 2]], cell=30)
            canvas = Image.new("RGB", (left.width + right.width + 70, max(left.height, right.height)), "white")
            canvas.paste(left, (0, 0))
            right_x = left.width + 70
            canvas.paste(right, (right_x, 0))
            draw = ImageDraw.Draw(canvas)
            y = canvas.height // 2
            draw.line((left.width + 8, y, left.width + 58, y), fill="#9d74c8", width=2)
            draw.polygon([(left.width + 58, y), (left.width + 50, y - 5), (left.width + 50, y + 5)], fill="#9d74c8")
            x = right_x + right.width - 18
            draw.line((x, right.height - 8, x, 8), fill="#9d74c8", width=2)
            draw.polygon([(x, 8), (x - 5, 16), (x + 5, 16)], fill="#9d74c8")
            manifest = make_manifest(tmp, [canvas])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            self.assertTrue(any(item["type"] == "connector" for item in parsed[0]["annotations"]))
            self.assertTrue(any(item["type"] == "arrow" for item in parsed[0]["annotations"]))

    def test_broken_partial_board_vertical_arrow_is_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            cell = 34
            gap = 4
            pad = 3
            image = draw_board([[0, 0, 1, 2], [3, 4, 5, 2]], cell=cell, gap=gap, pad=pad)
            draw = ImageDraw.Draw(image)
            x = pad + 3 * (cell + gap) + cell // 2
            y_top = pad + 5
            y_gap_top = pad + cell - 4
            y_gap_bottom = pad + cell + gap + 6
            y_bottom = pad + 2 * cell + gap - 5
            draw.line((x, y_top, x, y_gap_top), fill="#9d74c8", width=2)
            draw.line((x, y_gap_bottom, x, y_bottom), fill="#9d74c8", width=2)
            draw.polygon([(x, y_bottom), (x - 5, y_bottom - 8), (x + 5, y_bottom - 8)], fill="#9d74c8")
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            arrows = [item for item in parsed[0]["annotations"] if item["type"] == "arrow"]
            self.assertTrue(
                any(item.get("from_cell") == [0, 3] and item.get("to_cell") == [1, 3] for item in arrows)
            )

    def test_full_board_horizontal_arrows_keep_arrowhead_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            cell = 38
            gap = 4
            pad = 3
            image = draw_board(
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [6, 4, 3, 0]],
                cell=cell,
                gap=gap,
                pad=pad,
            )
            draw = ImageDraw.Draw(image)
            for row in (1, 2, 3):
                y = pad + row * (cell + gap) + cell // 2
                x0 = pad + 4
                x1 = pad + 4 * cell + 3 * gap - 4
                draw.line((x0, y, x1, y), fill="#9d74c8", width=2)
                draw.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill="#9d74c8")
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            arrows = [item for item in parsed[0]["annotations"] if item["type"] == "arrow"]
            for row in (1, 2, 3):
                self.assertTrue(
                    any(item.get("from_cell") == [row, 0] and item.get("to_cell") == [row, 3] for item in arrows)
                )

    def test_full_board_vertical_arrow_length_and_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            cell = 38
            gap = 4
            pad = 3
            image = draw_board(
                [[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 2], [4, 0, 0, 3]],
                cell=cell,
                gap=gap,
                pad=pad,
            )
            draw = ImageDraw.Draw(image)
            x = pad + 3 * (cell + gap) + cell // 2
            y0 = pad + 4
            y1 = pad + 4 * cell + 3 * gap - 4
            draw.line((x, y0, x, y1), fill="#9d74c8", width=2)
            draw.polygon([(x, y0), (x - 5, y0 + 8), (x + 5, y0 + 8)], fill="#9d74c8")
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            arrows = [item for item in parsed[0]["annotations"] if item["type"] == "arrow"]
            self.assertTrue(
                any(item.get("from_cell") == [3, 3] and item.get("to_cell") == [0, 3] for item in arrows)
            )

    def test_internal_red_text_residual_is_not_single_cell_highlight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            cell = 38
            gap = 4
            pad = 3
            image = draw_board(
                [[1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0], [4, 4, 4, 8]],
                cell=cell,
                gap=gap,
                pad=pad,
            )
            draw = ImageDraw.Draw(image)
            x0 = pad - 1
            y0 = pad - 1
            x1 = pad + 3 * cell + 2 * gap + 1
            y1 = pad + 4 * cell + 3 * gap + 1
            draw.rectangle((x0, y0, x1, y1), outline="#d42626", width=2)
            target_x = pad + 3 * (cell + gap) + cell // 2 - 13
            target_y = pad + 3 * (cell + gap) + cell // 2 - 11
            draw.text((target_x, target_y), "256", fill="#d42626", font=_font(14))
            manifest = make_manifest(tmp, [image])
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl")
            highlights = [item for item in parsed[0]["annotations"] if item["type"] == "highlight"]
            self.assertTrue(any(len(item.get("cells") or []) == 12 for item in highlights))
            self.assertFalse(any(item.get("cells") == [[3, 3]] for item in highlights))

    def test_override_replaces_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            manifest = make_manifest(tmp, [Image.new("RGB", (40, 40), "white")])
            overrides = tmp / "overrides.yaml"
            overrides.write_text(
                """
img_0000:
  status: accepted
  boards:
    - board_id: manual_b0
      bbox: [0, 0, 40, 40]
      visible_rows: 4
      visible_cols: 4
      hex: "11110000ffff2222"
      cells: []
      confidence: 1.0
      flags: [manual_override]
  annotations: []
""",
                encoding="utf-8",
            )
            parsed = parse_manifest(manifest, tmp / "parsed_images.jsonl", overrides_path=overrides)
            self.assertEqual(parsed[0]["status"], "accepted")
            self.assertEqual(parsed[0]["boards"][0]["hex"], "11110000ffff2222")
            self.assertTrue(parsed[0]["quality"]["used_override"])

    def test_extract_docx_media_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[1, 2, 3, 4]], cell=24)
            image_path = tmp / "image1.png"
            image.save(image_path)
            docx_path = tmp / "minimal.docx"
            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    <w:p><w:r><w:t>before text</w:t></w:r></w:p>
    <w:p><w:r><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p>
    <w:p><w:r><w:t>after text</w:t></w:r></w:p>
  </w:body>
</w:document>
""",
                )
                archive.writestr(
                    "word/_rels/document.xml.rels",
                    """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>
""",
                )
                archive.writestr("word/media/image1.png", image_path.read_bytes())

            records = extract_docx_media(docx_path, tmp / "out")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["context_before"], ["before text"])
            self.assertEqual(records[0]["context_after"], ["after text"])
            self.assertTrue(Path(records[0]["path"]).exists())

    def test_render_qa_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            manifest = make_manifest(tmp, [draw_board([[1, 2, 3, 4]])])
            parsed_path = tmp / "parsed_images.jsonl"
            parse_manifest(manifest, parsed_path)
            qa_path = tmp / "qa.html"
            render_qa_report(parsed_path, qa_path)
            qa_html = qa_path.read_text(encoding="utf-8")
            self.assertIn("Guide Image Parser QA", qa_html)
            self.assertIn("board-grid-wrap", qa_html)
            self.assertEqual(len(read_jsonl(parsed_path)), 1)

    def test_render_qa_report_uses_visible_rows_for_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            image = draw_board([[1, 2, 3, 4], [5, 6, 7, 8]])
            image_path = tmp / "image.png"
            image.save(image_path)
            parsed_path = tmp / "parsed_images.jsonl"
            cells = []
            for row in range(4):
                for col in range(4):
                    flags = ["padding_f"] if row >= 2 else []
                    cells.append(
                        {
                            "row": row,
                            "col": col,
                            "bbox": [col * 10, row * 10, 10, 10],
                            "value": 0,
                            "exponent": 15 if flags else 0,
                            "hex_digit": "f" if flags else "0",
                            "confidence": 1.0,
                            "flags": flags,
                        }
                    )
            record = {
                "image_id": "img_0000",
                "sha256": "sha",
                "source_name": "word/media/image1.png",
                "path": str(image_path),
                "relative_path": "media/image.png",
                "doc_index": 0,
                "width": image.width,
                "height": image.height,
                "status": "review",
                "boards": [
                    {
                        "board_id": "img_0000_b00",
                        "bbox": [0, 0, image.width, image.height],
                        "visible_rows": 2,
                        "visible_cols": 4,
                        "hex": "00000000ffffffff",
                        "cells": cells,
                        "confidence": 1.0,
                        "flags": [],
                    }
                ],
                "annotations": [
                    {
                        "type": "arrow",
                        "annotation_id": "a0",
                        "board_id": "img_0000_b00",
                        "from_cell": [0, 3],
                        "to_cell": [1, 3],
                        "bbox": [0, 0, 10, 20],
                        "confidence": 1.0,
                        "flags": [],
                    },
                    {
                        "type": "highlight",
                        "annotation_id": "h0",
                        "board_id": "img_0000_b00",
                        "cells": [[0, 3], [1, 3]],
                        "bbox": [0, 0, 10, 20],
                        "confidence": 1.0,
                        "flags": [],
                    }
                ],
                "quality": {},
            }
            parsed_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            qa_path = tmp / "qa.html"
            render_qa_report(parsed_path, qa_path)
            qa_html = qa_path.read_text(encoding="utf-8")
            self.assertIn("height:80px", qa_html)
            self.assertIn('viewBox="0 0 164 80"', qa_html)
            self.assertIn('x1="145.00" y1="19.00" x2="145.00" y2="61.00"', qa_html)
            self.assertIn('x="126.00" y="0.00" width="38.00" height="80.00"', qa_html)
            self.assertNotIn('y2="75.00%"', qa_html)


if __name__ == "__main__":
    unittest.main()
