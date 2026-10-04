import pytest

from backend.analysis_core import ReplayDecoder, bm, vbm


# Three records from a single-byte VRS: two initial tiles, then down + spawn.
VRS = b"3x3-1_\xbe\xfag\xbe\xfc0\xbe\xfa1"


@pytest.mark.parametrize("encoding", ["latin-1", "utf-8", "utf-8-sig", "utf-16"])
def test_verse_encoding_preserves_moves(tmp_path, encoding):
    path = tmp_path / "replay.vrs"
    path.write_bytes(VRS.decode("latin-1").encode(encoding))
    decoder = ReplayDecoder(str(path), bm, vbm)
    decoder.decode()
    assert decoder.variant == "3x3"
    assert len(decoder.record_list) == 1
    record = decoder.record_list[0]
    assert int(record["f0"]) == 0x010F000F100FFFFF
    assert int(record["f2"]) == 4
    assert int(record["f3"]) == 1
    assert int(record["f4"]) == 0


@pytest.mark.parametrize("payload", ["00000", "0000001", "000000\u260300"])
def test_invalid_verse_does_not_return_uninitialized_records(tmp_path, payload):
    path = tmp_path / "invalid.vrs"
    path.write_text("3x3-1_" + payload, encoding="utf-8")
    decoder = ReplayDecoder(str(path), bm, vbm)
    with pytest.raises(ValueError):
        decoder.decode()
