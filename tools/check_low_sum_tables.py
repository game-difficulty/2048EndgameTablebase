"""Read-only regression probe for early-layer AI handoff."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--native', required=True)
args = parser.parse_args()
dll_dirs = [os.add_dll_directory(str(ROOT / 'native_core')), os.add_dll_directory('C:/Apps/mingw64/bin')]
spec = importlib.util.spec_from_file_location('native_core.formation_core', args.native)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

import numpy as np
from Config import SingletonConfig
from engine_core.AIPlayer import DispatcherCommon
from engine_core.BookReader import BookReaderDispatcher
from engine_core.reader_results import complete_positive_moves

config = SingletonConfig().config
paths = [('C:/2048_tables/free10-256', 'uint32')]
config['filepath_map'] = {('free10_256', 0.1): paths}
config['4_spawn_rate'] = 0.1
reader = BookReaderDispatcher()
reader.dispatch(paths, 'free10', '256')
d = DispatcherCommon.__new__(DispatcherCommon)
d.book_reader = reader
d._table_cooldowns = {}
candidate = (14, 6, 6, 'free10', 8, '256', 'free10_256', 1, d._reader_state())
for code, accepted in [('101022109830edba', True), ('101202229813edba', False)]:
    board = np.array([2**int(c,16) if c != '0' else 0 for c in code]).reshape(4,4)
    d.reset(board, int(code,16))
    masked = d.mask(6)
    results, dtype = reader.move_on_dic(masked, 'free10', '256', 'free10_256')
    decision = d.check_table(candidate, 1)
    output = dict(board=code, small_sum=int(masked.sum())-6*32768,
                  results=results, dtype=dtype, legal_moves_mask=results.legal_moves_mask,
                  complete=complete_positive_moves(results), decision=decision)
    print(json.dumps(output), flush=True)
    assert (decision in ('left','right','up','down')) == accepted, output
