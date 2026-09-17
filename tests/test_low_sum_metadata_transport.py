import unittest
from unittest.mock import patch

from engine_core.reader_results import ReaderResults
from backend.gamer_tablebase_route import GamerRouteCursor, masked_board
from backend.tablebase_query_service import TablebaseQueryScheduler
from tools.tablebase_worker.protocol import sanitize_results


class CoverageTransportTests(unittest.TestCase):
    def test_metadata_survives_worker_stream_and_paid_cache(self):
        raw = ReaderResults({'left': .9, 'right': None, 'up': 0, 'down': .8}, 13)
        clean = sanitize_results(raw)
        cursor = GamerRouteCursor(dict(
            board_codes=[int(c,16) for c in '101022109830edba'],
            rng_state=[1,2,3,4], spawn_rate4=.1, difficulty=0,
            random_only=False, steps=1), 6)
        frame = cursor.node(clean, 'uint32')
        self.assertEqual(frame['legal_moves_mask'], 13)
        self.assertEqual(frame['lookup_board'], '10102210ff30ffff')
        # Reconstructing a remote frame must preserve metadata exactly.
        self.assertEqual(frame, cursor.node(dict(clean), 'uint32', 13))
        scheduler = TablebaseQueryScheduler(worker_count=1)
        try:
            with patch.object(scheduler, '_cache_set') as cache:
                result = scheduler.cache_stream_result(
                    catalog_version='coverage', full_pattern='free10_256',
                    board_encoded=cursor.encoded, results=frame['results'],
                    dtype=frame['dtype'], legal_moves_mask=frame['legal_moves_mask'])
                self.assertEqual(result.legal_moves_mask, 13)
                cache.assert_called_once()
                self.assertEqual(cursor.node(result.results, result.dtype, result.legal_moves_mask),frame)
        finally:
            scheduler._executor.shutdown()

    def test_legacy_results_do_not_claim_complete_coverage(self):
        clean = sanitize_results({'left': .9})
        self.assertIsNone(clean.legal_moves_mask)
        values = [2**int(c,16) if c!='0' else 0 for c in '101022109830edba']
        self.assertEqual(masked_board(values,6),int('10102210ff30ffff',16))
