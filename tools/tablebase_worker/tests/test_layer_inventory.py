import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.tablebase_worker.layer_inventory import LayerInventory
from backend.remote_workers.layers import normalize_layer_inventory


class LayerInventoryTests(unittest.TestCase):
    def test_split_paths_formats_holes_and_bc_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / 'a', Path(root) / 'b'
            first.mkdir(); second.mkdir()
            for name in ['0.bccmp', '1.bcraw', '4.bcpos', '5.bcpos', '7.exadbook', '8.exadzbook', '9.zbook', '10.exzbook', '11.book']:
                (first / ('free12_2048_' + name)).touch()
            (second / 'free12_2048_4.bcsuc').touch()
            (second / 'free12_2048_2.bcraw').touch()
            (second / 'free12_2048_12b').mkdir()
            (second / 'free12_2048_13.z').mkdir()
            (second / 'unrelated_3.book').touch()
            (second / 'free12_2048_6.bccmp.part').touch()
            table = SimpleNamespace(table_id='free12_2048', paths=(first, second))
            self.assertEqual(LayerInventory().get(table), {'version': 1, 'ranges': [[0, 2], [4, 4], [7, 13]]})

    def test_cache_refreshes_on_add_delete_and_unreadable_is_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            table = SimpleNamespace(table_id='free12_2048', paths=(path,))
            inventory = LayerInventory()
            layer = path / 'free12_2048_629.bcraw'
            layer.touch()
            initial = inventory.get(table)
            with patch.object(inventory, '_scan', side_effect=AssertionError('rescanned unchanged directory')):
                self.assertEqual(inventory.get(table), initial)
            stamp = path.stat().st_mtime_ns
            layer.unlink()
            os.utime(path, ns=(stamp + 1000000, stamp + 1000000))
            self.assertEqual(inventory.get(table)['ranges'], [])
            layer.touch()
            os.utime(path, ns=(stamp + 2000000, stamp + 2000000))
            self.assertEqual(inventory.get(table), initial)
            with patch.object(Path, 'stat', side_effect=PermissionError()):
                self.assertIsNone(inventory.get(table))

    def test_bad_metadata_is_ignored_not_treated_as_empty_coverage(self):
        for value in [None, {}, {'version': True, 'ranges': []}, {'version': 2, 'ranges': []},
                      {'version': 1, 'ranges': [[1, 0]]}, {'version': 1, 'ranges': [[0, 2], [2, 3]]},
                      {'version': 1, 'ranges': [[False, 2]]}, {'version': 1, 'ranges': [[0, 2**40]]}]:
            self.assertIsNone(normalize_layer_inventory(value))
        self.assertEqual(normalize_layer_inventory({'version': 1, 'ranges': []}), {'version': 1, 'ranges': []})
