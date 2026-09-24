"""Offline codec/layout comparison. Synthetic long traces are not claimed human games."""
import gzip
import json
import lzma
from pathlib import Path
import platform
import statistics
import time

from tools.measure_human_resources import synthetic, legal_game


def layouts(raw):
    varint = bytearray()
    for offset in range(0, len(raw), 5):
        varint.append(raw[offset])
        value = int.from_bytes(raw[offset + 1:offset + 5], 'little')
        while value >= 128:
            varint.append((value & 127) | 128); value >>= 7
        varint.append(value)
    return {'fixed5': raw, 'varint': bytes(varint), 'byte_planes': b''.join(raw[i::5] for i in range(5))}


def main():
    codecs = {'gzip6': (lambda b: gzip.compress(b, compresslevel=6, mtime=0), gzip.decompress),
              'gzip9': (lambda b: gzip.compress(b, compresslevel=9, mtime=0), gzip.decompress),
              'xz0': (lambda b: lzma.compress(b, preset=0), lzma.decompress),
              'xz6': (lambda b: lzma.compress(b, preset=6), lzma.decompress)}
    try:
        import brotli
        for level in (5, 9):
            codecs[f'brotli{level}'] = (lambda b, q=level: brotli.compress(b, quality=q), brotli.decompress)
    except ImportError:
        pass
    try:
        import zstandard as zstd
        for level in (3, 9):
            codecs[f'zstd{level}'] = (zstd.ZstdCompressor(level=level).compress, zstd.ZstdDecompressor().decompress)
    except ImportError:
        pass
    legal = max((legal_game(index)[1] for index in range(4)), key=len)
    samples = {'legal_seeded': legal, '100k_100_1000ms': synthetic(100000, (100, 1000)),
               '100k_150_3000ms': synthetic(100000, (150, 3000)),
               '100k_full_uint32': synthetic(100000, (0, 0xffffffff)),
               '200k_150_3000ms': synthetic(200000, (150, 3000))}
    rows = []
    for name, raw in samples.items():
        for layout, data in layouts(raw).items():
            for codec, (encode, decode) in codecs.items():
                # Warm up native codecs; report medians rather than cold-start noise.
                assert decode(encode(data)) == data
                enc_times, dec_times = [], []
                for _ in range(3):
                    start = time.perf_counter(); packed = encode(data); enc_times.append((time.perf_counter() - start) * 1000)
                    start = time.perf_counter(); restored = decode(packed); dec_times.append((time.perf_counter() - start) * 1000)
                    assert restored == data
                compressed_ms, decompressed_ms = statistics.median(enc_times), statistics.median(dec_times)
                rows.append(dict(sample=name, moves=len(raw)//5, layout=layout, codec=codec,
                                 raw_bytes=len(data), bytes=len(packed), encode_ms=round(compressed_ms, 3),
                                 decode_ms=round(decompressed_ms, 3)))
    result = {'platform': platform.platform(), 'python': platform.python_version(),
              'note': 'Median of 3 local timings after warmup; excludes container metadata; no production load test.', 'rows': rows}
    output = Path('output/human-codecs.json'); output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name in samples:
        group = [row for row in rows if row['sample'] == name]
        print(json.dumps({'sample': name, 'best': min(group, key=lambda row: row['bytes']),
                          'gzip': [row for row in group if row['codec'] == 'gzip6']}, ensure_ascii=False))
    print(output.resolve())


if __name__ == '__main__':
    main()
