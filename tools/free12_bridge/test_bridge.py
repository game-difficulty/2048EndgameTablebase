"""Native integration tests in a private temporary directory; no source writes."""
import argparse
from pathlib import Path
import struct
import tempfile
import unittest

import run


class BridgeIntegration(unittest.TestCase):
    def test_boundary_and_production_solve(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked), tempfile.TemporaryDirectory(
                    prefix="bridge-test-", dir=run.ROOT / "native_core/build-free12-bridge") as directory:
                args = argparse.Namespace(native=run.ROOT / "native_core/build-free12-bridge/free12_bridge.exe",
                                          threads=2, direct_io=False, chunked_solve=chunked)
                p = dict(prefix=str(Path(directory) / "free12_4096_"), signature=3107817247746329041,
                         stsl=96, seeds=[dict(target_step=1023), dict(target_step=1024)],
                         boundaries=[dict(target_step=1025), dict(target_step=1026)], solver_steps=1029)
                run.native(args, p, "fixtures", 0)
                run.solve(args, p)
                # Every legal successor is present: .9*.8 + .1*.7 = .79.
                path = run.artifact(p, 1024, ".exadbook")
                h = run.header(path)
                self.assertGreater(h["rows"], 0)
                self.assertEqual({s["width"] for s in h["active_slots"]}, {5})
                with path.open("rb") as stream:
                    stream.seek(-h["values"] * 4, 2)
                    values = struct.unpack(f"<{h['values']}I", stream.read())
                self.assertTrue(all(abs(v - 3160000000) <= 2 for v in values), (min(values), max(values)))
                # Re-enter solve after .exadtmp cleanup: all layers must skip.
                run.solve(args, p)
                self.assertFalse(run.artifact(p, 0, ".exadbook").exists())
                # The wrapper must reject a corrupt native checkpoint before resume.
                with path.open("r+b") as stream:
                    stream.truncate(path.stat().st_size - 1)
                with self.assertRaises(ValueError):
                    run.header(path)

    def test_actual_source_metadata_and_sum_mapping(self):
        if not Path("D:/free11-1k").exists() or not Path("G:/free11-2k").exists():
            self.skipTest("Local source datasets are unavailable")
        args = argparse.Namespace(source1=Path("D:/free11-1k"), source2=Path("G:/free11-2k"),
                                  output=Path("D:/free12-4k"), new_layers=64)
        p = run.plan(args)
        self.assertEqual(p["source_stsl"], [108, 96])
        self.assertEqual([s["target_step"] for s in p["seeds"]], [980, 981])
        self.assertEqual([b["target_step"] for b in p["boundaries"]], [1044, 1045])
        self.assertEqual([b["source_step"] for b in p["boundaries"]], [21, 22])
        self.assertLess(p["boundaries"][-1]["target_step"], p["solver_steps"] - 2)


if __name__ == "__main__":
    unittest.main()
