"""Table discovery/import uses filenames and metadata, never rewrites layer data."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Config import SingletonConfig, DTYPE_CONFIG
from backend.table_catalog import scan_tables, import_tables, catalog_snapshot


class TableCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(SingletonConfig().config, filepath_map={}, **{'4_spawn_rate': .1})
        self.cfg_patch = patch.object(SingletonConfig(), 'config', self.config)
        self.cfg_patch.start()
        self.addCleanup(self.cfg_patch.stop)
        self.save_patch = patch.object(SingletonConfig, 'save_config')
        self.save = self.save_patch.start()
        self.addCleanup(self.save_patch.stop)

    def result(self, prefix='3x3_sum-1800', suffix='.book', dtype='float64', rate='.1', sub=''):
        directory = self.root / sub
        directory.mkdir(exist_ok=True)
        (directory / (prefix + '_-1' + suffix)).write_bytes(b'result')
        (directory / (prefix + '_config.txt')).write_text(f'success_rate_dtype: {dtype}\n4_spawn_rate: {rate}\n')
        return directory

    def test_batch_import_is_idempotent_and_filters_probability(self):
        self.result(sub='classic')
        self.result('2x4_sum-900', '.zbook', rate='.2', sub='ex')
        rows = scan_tables(self.root)
        self.assertEqual(len(rows), 2)
        import_tables(rows)
        import_tables(rows)
        self.assertEqual(catalog_snapshot()['available_tables'], {'3x3': ['sum-1800']})
        self.assertTrue(all(len(entries) == 1 for entries in self.config['filepath_map'].values()))
        self.config['4_spawn_rate'] = .2
        self.assertEqual(catalog_snapshot()['available_tables'], {'2x4': ['sum-900']})

    def test_legacy_formats_and_all_precisions(self):
        for dtype in DTYPE_CONFIG:
            directory = self.result('L3_2048', '.exzbook', dtype=dtype, sub=dtype)
            self.assertEqual(scan_tables(directory)[0]['dtype'], dtype)
        directory = self.result('free12_4096', '.exadbook', sub='ad')
        (directory / 'free12_4096_config.txt').unlink()
        row = scan_tables(directory)[0]
        self.assertEqual(row['dtype'], 'uint32')
        self.assertTrue(row['assumed_rate'])
        self.assertFalse((directory / 'free12_4096_config.txt').exists())

    def test_temporary_files_and_unpaired_bc_are_excluded(self):
        self.result('L3_2048', '.bcpos')
        self.result('3x3_sum-1800', '.exadtmp')
        self.assertEqual(scan_tables(self.root), [])
        (self.root / 'L3_2048_-1.bcsuc').write_bytes(b'result')
        self.assertEqual(len(scan_tables(self.root)), 1)

    def test_metadata_disagreement_and_nan_rejected(self):
        self.result()
        metadata = self.root / '3x3_sum-1800_goal.json'
        metadata.write_text(json.dumps(dict(kind='sum', value=900)))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            scan_tables(self.root)
        metadata.unlink()
        self.result(rate='nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            scan_tables(self.root)

    def test_import_revalidates_before_any_registry_change(self):
        self.result()
        rows = scan_tables(self.root)
        rows[0]['dtype'] = 'uint32'
        self.assertEqual(import_tables(rows)[0]['dtype'], 'float64')
        self.config['filepath_map'].clear()
        rows.append(dict(rows[0], target='sum-900'))
        with self.assertRaises(ValueError):
            import_tables(rows)
        self.assertEqual(self.config['filepath_map'], {})

    def test_actions_broadcast_catalog_and_return_request_id(self):
        from backend.handlers.settings import handle_settings_action
        self.result()
        socket, manager = AsyncMock(), AsyncMock()
        asyncio.run(handle_settings_action('TABLE_IMPORT', {'request_id': 'test', 'tables': scan_tables(self.root)}, None, socket, manager))
        self.assertEqual(socket.send_json.call_args.args[0]['payload']['request_id'], 'test')
        event = json.loads(manager.broadcast.call_args.args[0])
        self.assertEqual(event['payload']['available_tables'], {'3x3': ['sum-1800']})

    def test_unregistered_sum_jump_retains_target_and_board(self):
        from backend.handlers.trainer import handle_trainer_action
        from backend.session import GameSession
        session = GameSession('catalog_jump')
        socket, manager = AsyncMock(), AsyncMock()
        board = '000f000f001fffff'
        asyncio.run(handle_trainer_action('TRAINER_LOAD_POSITION', {
            'full_pattern': '3x3_sum-1800', 'hex_str': board, 'request_id': 'jump'
        }, session, socket, manager))
        self.assertEqual(int(session.board_encoded), int(board, 16))
        self.assertEqual(session.current_pattern, '3x3_sum-1800')
        self.assertTrue(socket.send_json.call_args.args[0]['data']['success'])


if __name__ == '__main__':
    unittest.main()
