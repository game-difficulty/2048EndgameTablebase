"""Render the RIP gift's three-tile animation using the existing icon colors."""
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SIZE = (240, 100)
TILE = (68, 84)
STARTS = (10, 86, 162)


def frame(font_path, progress=0, spawn=0):
    image = Image.new('RGBA', SIZE)
    draw = ImageDraw.Draw(image)
    for x in STARTS:
        draw.rounded_rectangle((x, 8, x + TILE[0], 92), radius=10, fill='#273244')

    def tile(x, value, scale=1):
        width, height = (round(length * scale) for length in TILE)
        left, top = x + (TILE[0] - width) / 2, (SIZE[1] - height) / 2
        draw.rounded_rectangle((left, top, left + width, top + height),
                               radius=max(1, round(10 * scale)),
                               fill='#228b72' if value == 2 else '#f0d6a3')
        font = ImageFont.truetype(str(font_path), max(1, round(60 * scale)))
        text = str(value)
        bounds = draw.textbbox((0, 0), text, font=font)
        origin = (x + TILE[0] / 2 - (bounds[0] + bounds[2]) / 2,
                  SIZE[1] / 2 - (bounds[1] + bounds[3]) / 2)
        draw.text(origin, text, font=font, fill='white' if value == 2 else '#73592e')

    tile(STARTS[2], 4)
    tile(STARTS[0] + (STARTS[1] - STARTS[0]) * progress, 2)
    if spawn:
        tile(STARTS[0], 4, spawn)
    return image


def build(font_path, destination):
    frames, durations = [frame(font_path)], [300]
    for step in range(1, 13):
        progress = step / 12
        frames.append(frame(font_path, progress * progress * (3 - 2 * progress)))
        durations.append(40)
    durations[-1] += 80
    for scale in (.35, .7, 1.08, 1.04, 1):
        frames.append(frame(font_path, 1, scale))
        durations.append(40)
    durations[-1] += 1000

    # Use one palette and full-frame disposal so transparent moving tiles leave no trails.
    sheet = Image.new('RGB', (SIZE[0], SIZE[1] * len(frames)))
    for index, image in enumerate(frames):
        sheet.paste(image.convert('RGB'), (0, index * SIZE[1]))
    palette = sheet.quantize(colors=255)
    encoded = []
    for image in frames:
        indexed = image.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE)
        pixels = bytes(value + 1 if alpha else 0
                       for value, alpha in zip(indexed.tobytes(), image.getchannel('A').tobytes()))
        output = Image.frombytes('P', SIZE, pixels)
        output.putpalette([0, 0, 0] + palette.getpalette()[:765])
        encoded.append(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded[0].save(destination, save_all=True, append_images=encoded[1:],
                    duration=durations, loop=0, transparency=0, disposal=2, optimize=False)
    with Image.open(destination) as check:
        assert check.n_frames == len(frames)
        assert check.size == SIZE
        assert check.info['transparency'] == 0
        assert sum(check.seek(index) or check.info['duration'] for index in range(check.n_frames)) == sum(durations)
    print(f'{destination}: {len(frames)} frames, {sum(durations)} ms, {destination.stat().st_size} bytes')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True, help='Arial Bold TTF, matching GiftIcon.vue')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'frontend/public/live-gifts/rip-motion.gif')
    args = parser.parse_args()
    build(args.font, args.output)
