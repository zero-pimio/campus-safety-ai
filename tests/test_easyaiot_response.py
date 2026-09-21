import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campus_safety_ai.adapters.easyaiot import EasyAIoTDeliveryError, EasyAIoTDestination, _post_json
from campus_safety_ai.core.event_delivery import EventDelivery
import test_easyaiot_adapter


class Response(io.BytesIO):
    status = 200


class EasyAIoTResponseTests(unittest.TestCase):
    def test_only_explicit_upstream_acceptance_succeeds(self):
        cases = [
            ({"code": 0, "data": {"status": "success", "alert_id": 1}}, True),
            ({"code": 500, "data": {"status": "failed"}}, False),
            ({"code": 0, "data": {"status": "skipped"}}, False),
            ({"code": 0, "data": {"status": "suppressed"}}, False),
            ({"code": 0, "data": None}, False),
            ({"code": False, "data": {"status": "success"}}, False),
            ({"code": 0, "data": {"status": []}}, False),
            ({"code": 200, "data": {"status": "success"}}, False),
            ([], False),
        ]
        for payload, accepted in cases:
            with self.subTest(payload=payload), patch(
                "campus_safety_ai.adapters.easyaiot.urlopen",
                return_value=Response(json.dumps(payload).encode()),
            ):
                if accepted:
                    self.assertEqual(_post_json("http://localhost/hook", b"{}", {}, 1), 200)
                else:
                    with self.assertRaises(EasyAIoTDeliveryError):
                        _post_json("http://localhost/hook", b"{}", {}, 1)

    def test_invalid_and_oversized_bodies_are_rejected(self):
        for raw in (b"<html>login</html>", b"x" * 1_048_577):
            with self.subTest(size=len(raw)), patch(
                "campus_safety_ai.adapters.easyaiot.urlopen", return_value=Response(raw)
            ), self.assertRaises(EasyAIoTDeliveryError):
                _post_json("http://localhost/hook", b"{}", {}, 1)

    def test_http_200_skipped_event_remains_pending(self):
        record = test_easyaiot_adapter.EasyAIoTAdapterTests().record()
        with tempfile.TemporaryDirectory() as directory:
            delivery = EventDelivery(
                Path(directory) / "outbox.sqlite3", EasyAIoTDestination("http://localhost/hook")
            )
            try:
                with patch("campus_safety_ai.adapters.easyaiot.urlopen", return_value=Response(
                    b'{"code":0,"data":{"status":"skipped"}}'
                )), self.assertRaises(EasyAIoTDeliveryError):
                    delivery.submit([record])
                self.assertEqual(delivery.pending_count(), 1)
            finally:
                delivery.close()
