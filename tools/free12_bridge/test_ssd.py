"""Small production-solver tests; never touch the live bridge output."""
import argparse
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import run
import ssd


class SSDTests(unittest.TestCase):
    def test_hot_cold_solve_archive_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="ssd-test-", dir=run.ROOT / "native_core/build-free12-bridge") as tmp:
            root = Path(tmp)
            cold = root / "cold"
            cold.mkdir()
            p = dict(output=str(cold), prefix=str(cold / "free12_4096_"),
                     source1=str(root / "source1" / "free11_1024_"),
                     source2=str(root / "source2" / "free11_2048_"),
                     signature=3107817247746329041, stsl=96,
                     seeds=[dict(target_step=1023), dict(target_step=1024)],
                     boundaries=[dict(target_step=1025), dict(target_step=1026)], solver_steps=1029)
            args = argparse.Namespace(native=run.ROOT / "native_core/build-free12-bridge/free12_bridge.exe",
                                      threads=2, direct_io=False, chunked_solve=True, reserve_gib=1)
            run.native(args, p, "fixtures", 0)
            hp = ssd.hot_plan(p, root / "hot")
            self.assertEqual(ssd.inspect(p, hp), 1024)
            ssd.stage(p, hp, 1024, ssd.GIB)
            archive = ssd.Archiver(p, hp, 1024)
            # Still-needed futures must stay on SSD before any layer completes.
            archive.retire(1027)
            self.assertTrue(run.artifact(hp, 1025, ".exadbook").exists())
            ssd.solve_ssd(args, p, hp)
            self.assertEqual(ssd.completed_step(hp), 1023)
            self.assertIsNone(ssd.inspect(p, hp))
            for step in range(1023, 1027):
                self.assertTrue(run.artifact(p, step, ".exadbook").exists())
                self.assertFalse(run.artifact(hp, step, ".exadbook").exists())
            path = run.artifact(p, 1024, ".exadbook")
            h = run.header(path)
            with path.open("rb") as stream:
                stream.seek(-h["values"] * 4, 2)
                values = struct.unpack(f"<{h['values']}I", stream.read())
            self.assertTrue(values)
            self.assertTrue(all(abs(v - 3160000000) <= 2 for v in values))
            # Completed run: stage lowest two for final pruning, then archive again.
            ssd.solve_ssd(args, p, hp)
            self.assertFalse(run.artifact(hp, 0, ".exadbook").exists())
            # An interrupted transfer must leave the source and old destination intact.
            dest = root / "copy.exadbook"
            dest.write_bytes(b"old")
            with patch.object(ssd.os, "replace", side_effect=OSError("injected failure")):
                with self.assertRaises(OSError):
                    ssd.copy_publish(path, dest, solved=True, remove_source=True)
            self.assertTrue(path.exists())
            self.assertEqual(dest.read_bytes(), b"old")
            ssd.copy_publish(path, dest, solved=True)
            self.assertEqual(dest.read_bytes(), path.read_bytes())

    def test_output_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with run.lock_output(Path(directory)):
                with self.assertRaises(OSError):
                    with run.lock_output(Path(directory)):
                        self.fail("second writer acquired the lock")


if __name__ == "__main__":
    unittest.main()
