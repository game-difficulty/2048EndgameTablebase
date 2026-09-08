import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Config import pattern_catalog, pattern_32k_tiles_map
from backend import tablebase_catalog as catalog
from backend.gamer_tablebase_route import masked_board
from backend.trainer_helpers import replace_largest_tiles


ROOT = Path(__file__).resolve().parents[1]


class GamerTableStructureTests(unittest.TestCase):
    def test_metadata_uses_native_masks_and_versions_catalog(self):
        entry = dict(pattern='LL', target='1024', _full_pattern='LL_1024')
        metadata = catalog.ai_table_metadata(entry)
        self.assertEqual(metadata['structure'], dict(version=1, transforms='dihedral8',
            pattern_masks=[f'{int(mask):016x}' for mask in pattern_catalog['LL']['pattern_masks']]))
        self.assertNotIn('structure', catalog.ai_table_metadata(dict(pattern='3x3')))
        self.assertNotIn('structure', catalog.ai_table_metadata(dict(pattern='unknown')))
        with patch.object(catalog, '_iter_available_entries', return_value=[entry]), \
                patch.object(catalog, '_iter_local_entries', return_value=[]), \
                patch.object(catalog, '_CATALOG_VERSION_CACHE', (0, '', -1)):
            before = catalog.get_catalog_version()
            with patch.dict(pattern_catalog, LL={**pattern_catalog['LL'], 'pattern_masks': (0,)}), \
                    patch.object(catalog, '_CATALOG_VERSION_CACHE', (0, '', -1)):
                self.assertNotEqual(before, catalog.get_catalog_version())

    def test_js_filter_against_native_predicate_and_server_masking(self):
        compiler = shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('C++ compiler required for native is_pattern parity')
        rng = random.Random(90218)
        metadata, cases, probe_lines, expected = [], [], [], []
        for pattern in pattern_32k_tiles_map:
            ai = catalog.ai_table_metadata(dict(pattern=pattern))
            if not ai['compatible']:
                continue
            rule = len(metadata)
            metadata.append(ai['structure'])
            count = ai['large_tiles']
            boards = []
            # Seeds exercise every accepted geometry; random boards include ties and >32K tiles.
            for seed in pattern_catalog[pattern]['seed_boards']:
                codes = np.array([int(char, 16) for char in f'{int(seed):016x}']).reshape(4, 4)
                for turn in range(4):
                    rotated = np.rot90(codes, turn)
                    for mirrored in [rotated, np.fliplr(rotated)]:
                        boards.append([2 ** int(code) if code else 2 for code in mirrored.flat])
            for _ in range(40):
                codes = rng.sample(range(1, 19), 12) + [rng.randrange(12) for _ in range(4)]
                rng.shuffle(codes)
                boards.append([2 ** code if code else 0 for code in codes])
            boards.extend([[32768] * 16, [2] * 16,
                           [65536, 32768, 32768] + [2] * 13,
                           [131072, 65536] + [2] * 14])
            masks = [int(mask, 16) for mask in ai['structure']['pattern_masks']]
            for board in boards:
                # Follow both production masking stages, including numpy's tie selection.
                encoded = int(replace_largest_tiles(masked_board(board, count), count, '1024'))
                matrix = np.array([int(char, 16) for char in f'{encoded:016x}']).reshape(4, 4)
                transforms = []
                for turn in range(4):
                    rotated = np.rot90(matrix, turn)
                    for mirrored in [rotated, np.fliplr(rotated)]:
                        transforms.append(int(''.join(f'{int(code):x}' for code in mirrored.flat), 16))
                expected.append('match' if not masks or any(
                    value & mask == mask for value in transforms for mask in masks) else 'mismatch')
                probe_lines.append(' '.join([str(len(masks)),
                    *(f'{mask:x}' for mask in masks), *(f'{value:x}' for value in transforms)]))
                cases.append(dict(board=board, count=count, rule=rule))
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'table_structure_probe.exe'
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(ROOT / 'native_core/include'),
                str(ROOT / 'tests/native/table_structure_probe.cpp'), '-o', str(executable)],
                check=True, capture_output=True, text=True)
            native = subprocess.run([str(executable)], input='\n'.join(probe_lines) + '\n',
                                    capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(native, expected)
        result = subprocess.run(['node', str(ROOT / 'frontend/tests/tableStructureRunner.mjs')],
            input=json.dumps(dict(metadata=metadata, cases=cases)), capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
        self.assertEqual(len(actual), len(native))
        for index, (js, cpp) in enumerate(zip(actual, native)):
            self.assertIn(js, ['match', 'mismatch', 'unknown'])
            if js != 'unknown':
                self.assertEqual(js, cpp, f'case={cases[index]}')
        self.assertGreater(actual.count('match'), 0)
        self.assertGreater(actual.count('mismatch'), 0)
        self.assertGreater(actual.count('unknown'), 0)
        self.assertLess(actual.count('unknown'), len(actual) // 2)


if __name__ == '__main__':
    unittest.main()
