"""Blend the 2048 illustration with a real QR matrix into a static image.

Build-time requirements: ``pip install qrcode pillow``. The poster only loads
the generated WebP; it never calls an external QR service or image model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import qrcode
from qrcode.util import pattern_position
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/assets/play-qr-art-source.png"
CUTOUT = ROOT / "tools/assets/play-qr-art-cutout.png"
DEFAULT_OUTPUT = ROOT / "frontend/public/human/play-qr-art.webp"
DEFAULT_BACKDROP = ROOT / "frontend/public/human/play-qr-backdrop.webp"
URL = "https://play.2048tables.online/"
INK = "#24132d"
PAPER = "#aa8fa0"
MODULE_SIZE = 18


def render(output: Path, backdrop_output: Path, dark_strength: float) -> None:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=4)
    qr.add_data(URL)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    side = len(matrix)
    quiet = 4
    image_size = side * MODULE_SIZE
    alignment_positions = pattern_position(qr.version)

    art = Image.open(SOURCE).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    dark_layer = Image.blend(art, Image.new("RGB", art.size, INK), dark_strength)
    light_layer = Image.blend(art, Image.new("RGB", art.size, PAPER), .85).convert("RGBA")
    light_layer.putalpha(175)
    quiet_layer = Image.blend(art, Image.new("RGB", art.size, PAPER), .85).convert("RGBA")
    result = Image.new("RGBA", art.size, (0, 0, 0, 0))
    pure_dark = Image.new("RGB", (MODULE_SIZE, MODULE_SIZE), INK)
    pure_light = Image.new("RGB", (MODULE_SIZE, MODULE_SIZE), PAPER)

    def function_module(col: int, row: int) -> bool:
        # Finder eyes and their immediate separators; timing and the version-4
        # alignment eye retain maximal contrast. The data modules carry art.
        if any(x <= col < x + 8 and y <= row < y + 8 for x, y in
               ((quiet - 1, quiet - 1), (side - quiet - 7, quiet - 1),
                (quiet - 1, side - quiet - 7))):
            return True
        if row == quiet + 6 or col == quiet + 6:
            return True
        for center_row in alignment_positions:
            for center_col in alignment_positions:
                last = alignment_positions[-1]
                if (center_col, center_row) in ((6, 6), (6, last), (last, 6)):
                    continue
                if (abs(col - (quiet + center_col)) <= 2
                        and abs(row - (quiet + center_row)) <= 2):
                    return True
        return False

    for row, bits in enumerate(matrix):
        for col, bit in enumerate(bits):
            x, y = col * MODULE_SIZE, row * MODULE_SIZE
            box = (x, y, x + MODULE_SIZE, y + MODULE_SIZE)
            if row < quiet or col < quiet or row >= side - quiet or col >= side - quiet:
                distance = min(row, col, side - 1 - row, side - 1 - col)
                patch = quiet_layer.crop(box)
                patch.putalpha([0, 65, 145, 220][distance])
                result.paste(patch, (x, y))
                continue
            if function_module(col, row):
                patch = pure_dark if bit else pure_light
            elif bit:
                patch = dark_layer.crop(box)
            else:
                patch = light_layer.crop(box)
            result.paste(patch, (x, y))

    # The separately extracted illustration has real alpha and an irregular
    # silhouette. Keep that silhouette intact; no rectangular fill or radial
    # vignette is baked into the asset.
    backdrop = Image.open(CUTOUT).convert("RGBA").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    backdrop.putalpha(backdrop.getchannel("A").point(lambda value: round(value * .72)))
    output.parent.mkdir(parents=True, exist_ok=True)
    backdrop_output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".webp":
        result.save(output, format="WEBP", quality=95, method=6, exact=True)
    else:
        result.save(output, format="PNG", optimize=True)
    backdrop.save(backdrop_output, format="WEBP", quality=85, method=6, exact=True)
    print(f"Wrote {output} ({side}x{side} modules, {image_size}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backdrop-output", type=Path, default=DEFAULT_BACKDROP)
    parser.add_argument("--dark-strength", type=float, default=.8)
    args = parser.parse_args()
    if not 0 <= args.dark_strength <= 1:
        parser.error("blend strength must be between 0 and 1")
    render(args.output, args.backdrop_output, args.dark_strength)
