# Guide Image Parser

Offline tools for converting Word guide images into structured 2048 board data.
The tools are intentionally outside the cloud runtime path; they are for asset
preparation and QA only.

## Commands

```powershell
python tools/guide_image_parser/extract_docx_media.py --docx "C:\path\guide.docx" --out tmp/guide_media
python tools/guide_image_parser/parse_guide_images.py --manifest tmp/guide_media/media_manifest.jsonl --out tmp/guide_media/parsed_images.jsonl --qa-dir tmp/guide_media/qa
python tools/guide_image_parser/validate_gold_samples.py --parsed tmp/guide_media/parsed_images.jsonl --gold tools/guide_image_parser/gold_samples.yaml
python tools/guide_image_parser/render_qa_report.py --parsed tmp/guide_media/parsed_images.jsonl --out tmp/guide_media/qa.html
python tools/guide_image_parser/build_guide_document.py --docx "C:\path\guide.docx" --manifest tmp/guide_media/media_manifest.jsonl --parsed tmp/guide_media/parsed_images.jsonl --out frontend/public/guides/my-guide --document-id my-guide --title "My Guide" --index frontend/public/guides/index.json
```

`parse_guide_images.py` accepts `--overrides tools/guide_image_parser/overrides.yaml`
for manual corrections keyed by `image_id` or `sha256`.

Run `validate_gold_samples.py` before asking for manual review. The gold file is
a small set of manually checked real guide images that catches obvious parser
regressions in board values and highlight mapping.

Gold samples may either list every board in spatial order, or target corrections
with `board_id`/`board_index`. Targeted entries can independently check `hex`,
`bbox` with an optional `bbox_tolerance`, and visible dimensions. Use
`board_count` when only the number of boards is known.

## Output Contract

- `media_manifest.jsonl`: one line per image occurrence in document order.
- `parsed_images.jsonl`: one line per image with detected boards, annotations,
  status, and QA flags.
- `guide_boards.json`: compact summary of accepted/review boards for later
  guide rendering.
- `qa.html`: original image plus parser overlay, expanded by default for review
  entries.
- `document.json`: browser-ready headings, paragraphs, original-image references,
  and clickable board hotspots. The optional guide index makes additional
  documents appear in the Help document selector.

v1 prioritizes correctness over automatic acceptance. Low-confidence cells,
color/text conflicts, unmatched annotation residue, or uncertain layout produce
`status: "review"`.
