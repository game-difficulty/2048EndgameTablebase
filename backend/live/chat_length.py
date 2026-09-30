"""Shared weighted chat length: Latin 1, CJK/emoji 2, joined emoji once."""
LIMIT = 80


def chat_length(text):
    total = 0
    joined = False
    regional = False
    for char in text:
        n = ord(char)
        if n == 0x200D:
            joined = True
            continue
        if (0x300 <= n <= 0x36F or 0xFE00 <= n <= 0xFE0F
                or 0x1F3FB <= n <= 0x1F3FF or 0xE0100 <= n <= 0xE01EF
                or n == 0x20E3):
            continue
        flag = 0x1F1E6 <= n <= 0x1F1FF
        if joined or (flag and regional):
            joined = False
            regional = False
            continue
        regional = flag
        wide = (0x1100 <= n <= 0x11FF or 0x2600 <= n <= 0x27BF
                or 0x2E80 <= n <= 0xA4CF or 0xAC00 <= n <= 0xD7AF
                or 0xF900 <= n <= 0xFAFF or 0xFE10 <= n <= 0xFE6F
                or 0xFF01 <= n <= 0xFF60 or n >= 0x1F000)
        total += 2 if wide else 1
    return total
