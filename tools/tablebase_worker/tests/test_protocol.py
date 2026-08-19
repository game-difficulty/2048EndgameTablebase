from __future__ import annotations

import json
import unittest

from tools.tablebase_worker.protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    validate_hello_ack,
    validate_request,
)


TABLES = {"free11_512"}
META = {"free11_512": ("free11", "512")}


def request(**overrides):
    value = {
        "type": "LOOKUP",
        "request_id": "request-1",
        "full_pattern": "free11_512",
        "pattern": "free11",
        "target": "512",
        "board": "0011223344556677",
        "use_variant": False,
        "board_is_lookup": False,
    }
    value.update(overrides)
    return value


class WorkerProtocolTests(unittest.TestCase):
    def validate(self, message):
        return validate_request(
            message,
            allowed_tables=TABLES,
            table_metadata=META,
            max_batch_size=4,
        )

    def test_hello_is_flat(self):
        encoded = encode_message(
            "HELLO",
            protocol_version=1,
            worker_id="home-main",
            auth_token="secret",
            tables=[{"full_pattern": "free11_512", "ready": True}],
        )
        message = json.loads(encoded)
        self.assertNotIn("data", message)
        self.assertEqual(message["type"], "HELLO")
        self.assertTrue(message["tables"][0]["ready"])

    def test_accepts_exact_server_hello_ack(self):
        ack = encode_message(
            "HELLO_ACK",
            protocol_version=1,
            worker_id="home-main",
            heartbeat_timeout_seconds=30,
            tables=["free11_512"],
        )
        parsed = validate_hello_ack(ack, worker_id="home-main")
        self.assertEqual(parsed["heartbeat_timeout_seconds"], 30)

    def test_heartbeat_is_flat_and_carries_readiness(self):
        tables = [{"full_pattern": "free11_512", "ready": True}]
        self.assertEqual(
            json.loads(encode_message("HEARTBEAT", tables=tables)),
            {"type": "HEARTBEAT", "tables": tables},
        )

    def test_lookup_parses_board_and_flags(self):
        parsed = self.validate(request(board_is_lookup=True))
        self.assertEqual(parsed.boards, (0x0011223344556677,))
        self.assertTrue(parsed.board_is_lookup)

    def test_lookup_batch_preserves_order(self):
        message = request(
            type="LOOKUP_BATCH",
            boards=["0000000000000001", "0000000000000002"],
        )
        message.pop("board")
        parsed = self.validate(message)
        self.assertEqual(parsed.boards, (1, 2))

    def test_random_state_schema(self):
        parsed = self.validate(
            {
                "type": "RANDOM_STATE",
                "request_id": "random-1",
                "full_pattern": "free11_512",
                "pattern": "free11",
                "target": "512",
            }
        )
        self.assertEqual(parsed.message_type, "RANDOM_STATE")

    def test_cancel_schema(self):
        parsed = self.validate({"type": "CANCEL", "request_id": "request-1"})
        self.assertEqual(parsed.message_type, "CANCEL")

    def test_rejects_remote_path_field(self):
        with self.assertRaisesRegex(ProtocolError, "unsupported"):
            self.validate(request(path="D:/replacement"))

    def test_rejects_unlisted_table(self):
        with self.assertRaisesRegex(ProtocolError, "allowlisted"):
            self.validate(
                request(
                    full_pattern="free12_2048",
                    pattern="free12",
                    target="2048",
                )
            )

    def test_rejects_pattern_target_mismatch(self):
        with self.assertRaisesRegex(ProtocolError, "local allowlist"):
            self.validate(request(pattern="free10"))

    def test_rejects_non_boolean_lookup_flags(self):
        with self.assertRaisesRegex(ProtocolError, "boolean"):
            self.validate(request(board_is_lookup=1))

    def test_rejects_malformed_board(self):
        with self.assertRaisesRegex(ProtocolError, "16 hexadecimal"):
            self.validate(request(board="1234"))

    def test_rejects_oversized_batch(self):
        message = request(
            type="LOOKUP_BATCH",
            boards=[f"{index:016x}" for index in range(5)],
        )
        message.pop("board")
        with self.assertRaisesRegex(ProtocolError, "configured limit"):
            self.validate(message)

    def test_decode_rejects_non_json(self):
        with self.assertRaisesRegex(ProtocolError, "valid JSON"):
            decode_message("not-json")


if __name__ == "__main__":
    unittest.main()
