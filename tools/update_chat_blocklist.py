"""Regenerate the chat deny list from a pinned, licensed upstream snapshot."""

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd967c30b053fa40b06c5a0dddf0be493f2dfae46'
BASE = f'https://raw.githubusercontent.com/konsheng/Sensitive-lexicon/{COMMIT}/Vocabulary/'
SOURCES = {
    '反动词库.txt': 4,
    '暴恐词库.txt': 4,
    '涉枪涉爆.txt': 4,
    '色情词库.txt': 4,
    '色情类型.txt': 3,
    '广告类型.txt': 4,
}


def normalized(term):
    return ''.join(char for char in unicodedata.normalize('NFKC', term).casefold()
                   if not unicodedata.category(char).startswith(('Z', 'C', 'P')))


def main():
    legacy = ROOT / 'docs_and_configs' / 'chat_blocklist_legacy.json'
    words = set(json.loads(legacy.read_text(encoding='utf-8')))
    for filename, minimum_length in SOURCES.items():
        with urlopen(BASE + quote(filename), timeout=20) as response:
            content = response.read().decode('utf-8-sig')
        for raw in re.split(r'[,，\r\n]+', content):
            word = raw.strip()
            compact = normalized(word)
            if (minimum_length <= len(compact) <= 32
                    and any('\u4e00' <= char <= '\u9fff' for char in compact)):
                words.add(word)
    target = ROOT / 'docs_and_configs' / 'chat_blocklist.json'
    target.write_text(json.dumps(sorted(words), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(words)} terms -> {target}')


if __name__ == '__main__':
    main()
