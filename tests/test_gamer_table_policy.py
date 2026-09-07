import json
import random
import subprocess
import unittest
import importlib.util
from pathlib import Path

import numpy as np
from Config import category_info, pattern_32k_tiles_map
from engine_core.VBoardMover import encode_board
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.gamer_ranked.rules import random_spawn


class GamerTablePolicyParityTests(unittest.TestCase):
    def test_python_js_decisions_masks_and_rng(self):
        desktop = Path(__file__).resolve().parents[2] / 'src/engine_core/AIPlayer.py'
        if not desktop.is_file():
            self.skipTest('Desktop DispatcherCommon is required for the cross-repository parity audit')
        spec = importlib.util.spec_from_file_location('desktop_ai_policy', desktop)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DispatcherCommon = module.DispatcherCommon
        rng = random.Random(5218)
        tables = []
        readers = {}
        for pattern, (n, free, fixed) in pattern_32k_tiles_map.items():
            if pattern in category_info.get('variant', []) or '_' in pattern:
                continue
            for target in [128, 256, 512, 1024, 2048]:
                exp = target.bit_length() - 1
                full = f'{pattern}_{target}'
                tables.append(dict(pattern=pattern, target=str(target), fullPattern=full, spawnRate=0.1,
                    ai=dict(compatible=True, policy_version=1, large_tiles=n, free_tiles=free)))
                readers.setdefault((n + exp, n), []).append((n + exp,n,free,pattern,exp,str(target),full,len(tables),None))
        for group in readers.values():
            group.sort(key=lambda row: row[2], reverse=True)
        cases = []; expected = []
        for _ in range(300):
            codes = rng.sample(list(range(1, 19)), 10) + [rng.randrange(7) for _ in range(6)]
            rng.shuffle(codes)
            values = [2 ** code if code else 0 for code in codes]
            dispatcher = DispatcherCommon.__new__(DispatcherCommon)
            dispatcher.board = np.asarray(values, dtype=np.int32).reshape(4, 4)
            packed = ''.join(format(min(code, 15), 'x') for code in codes)
            dispatcher.board_encoded = int(packed, 16)
            dispatcher.counts = dispatcher.frequency_count()
            dispatcher.ad_readers = readers
            dispatcher._table_cooldowns = {}
            groups = dispatcher.get_endgame_lvls()
            candidates = [(row, index + 1) for index, group in enumerate(groups) for row in group]
            probes = []; decisions = []
            for row, kind in candidates[:5]:
                for value,dtype in [(0.0,'uint32'),(.5,'uint32'),(1.0,'uint32'),(-.01,'1-float64')]:
                    probes.append(dict(table=dict(target=row[5], fullPattern=row[6]),type=kind,value=value,dtype=dtype))
                    dispatcher._table_cooldowns.clear()
                    dispatcher._restore_reader_state = lambda state: None
                    dispatcher.book_reader = type('Reader', (), {'move_on_dic': lambda *args: ({'left': value},dtype)})()
                    decisions.append(dispatcher.check_table(row, kind))
            cases.append(dict(board=values, probes=probes))
            expected.append(dict(candidates=[[row[6],kind] for row,kind in candidates],
                masks=[f'{int(encode_board(dispatcher.mask(row[1]))):016x}' for row,kind in candidates],
                decisions=decisions))
        spawns = []; spawn_expected = []
        for difficulty in [0,1,25,50,99,100]:
            for random_only in [False,True]:
                for _ in range(20):
                    state = [rng.randrange(2 ** 32) for _ in range(4)]
                    values = [0,2,0,4]*4
                    options = dict(difficulty=difficulty,randomOnly=random_only,spawnRate4=.3)
                    spawns.append(dict(board=values,state=state,options=options))
                    source = Xoshiro128StarStar(state.copy())
                    branch = None if random_only else source.next_float()
                    evil = not random_only and (difficulty>=100 or (difficulty>0 and branch<difficulty/100))
                    spawn = None
                    if not evil:
                        index,exp = random_spawn(values,source,.3)
                        spawn = dict(index=index,value=2 ** exp)
                    spawn_expected.append(dict(evil=evil,spawn=spawn,state=source.state))
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(['node', str(root/'frontend/tests/gamerTablePolicyRunner.mjs')],
            input=json.dumps(dict(tables=tables,cases=cases,spawns=spawns)),text=True,capture_output=True,check=True)
        actual = json.loads(completed.stdout)
        for index, (left, right) in enumerate(zip(actual['results'], expected)):
            for field in ['candidates','masks','decisions']:
                self.assertEqual(left[field], right[field], f'case={index} field={field}')
        self.assertEqual(actual['spawns'],spawn_expected)


if __name__ == '__main__':
    unittest.main()
