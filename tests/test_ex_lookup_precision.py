"""Exercise real EX readers, not mocked analysis rates.

Set EX_PRECISION_TABLE_DIR to an installed 3x3 sum-1790 table directory.
Only a tiny copied layer is edited; the original table is never modified.
"""
import importlib.util
import math
import os
from pathlib import Path
import struct

import pytest

HEADER = struct.Struct("<8s5I2BH9Q")
BOARD = 0x000F000F020FFFFF


@pytest.fixture(scope="module")
def native():
    module_path = os.environ.get("EX_PRECISION_NATIVE_MODULE")
    if module_path:
        spec = importlib.util.spec_from_file_location("formation_core", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from native_core import formation_core
    return formation_core


@pytest.fixture(scope="module")
def table():
    root = Path(os.environ.get(
        "EX_PRECISION_TABLE_DIR", "C:/2048_tables/cloud tables/3x3-sum1790"))
    layer = root / "3x3_sum-1790_2.zbook"
    lut = root / "3x3_sum-1790_.zlut"
    if not layer.is_file() or not lut.is_file():
        pytest.skip("EX_PRECISION_TABLE_DIR must contain a sum-1790 prefix36 fixture")
    return layer.read_bytes(), lut


@pytest.mark.parametrize("mode,kind,fmt,value,scale", [
    (0, 0, "I", 145217, 4e9),
    (1, 1, "Q", 58086812631415, 1.6e18),
    (2, 2, "f", 0.000036304257, 1.0),
    (3, 3, "d", 0.0000363042578690716, 1.0),
    (4, 2, "f", -0.000036304257, 1.0),
    (5, 3, "d", -0.0000363042578690716, 1.0),
])
def test_prefix36_preserves_stored_success_bits(
    native, table, tmp_path, mode, kind, fmt, value, scale
):
    data, lut = table
    header = list(HEADER.unpack_from(data))
    assert header[0] == b"EXP36BK\x00" and header[1] == 5
    offset = HEADER.size + header[11] * 16 + header[12] + header[13] * 8
    packed = struct.pack("<" + fmt, value)
    header[2], header[5], header[16] = kind, mode, len(packed)
    layer = tmp_path / "probe.zbook"
    layer.write_bytes(HEADER.pack(*header) + data[HEADER.size:offset] + packed * header[14])
    expected_bits = int.from_bytes(packed, "little")
    expected_value = struct.unpack("<" + fmt, packed)[0] / scale
    compressed = tmp_path / "probe.exzbook"
    native.compress_ex_zbook_result(str(layer), str(lut), str(compressed))
    for lookup, path in (
        (native.lookup_ex_zbook_cold, layer),
        (native.lookup_ex_zbook_result_cold, compressed),
    ):
        result = lookup(str(path), str(lut), BOARD)
        assert result["found"]
        assert result["success_kind"] == kind
        assert result["raw_value_bits"] == expected_bits
        assert math.isclose(result["numeric_value"], expected_value, rel_tol=2e-16, abs_tol=0.0)
