import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "parking_cv"))
from supabase_delivery import SupabaseDelivery, model_spot_ids
from watch_snapshots import FolderWatcher


class TestAnalyzer:
    calibration_id = "test-calibration"
    mapping = {"lot": "front", "spaces": [
        {"id": "B019"}, {"id": "B020", "space_type": "accessible"}, {"id": "B021"},
    ]}

    def __init__(self):
        self.calls = 0

    def analyze(self, image):
        self.calls += 1
        return {"occupied": 1, "estimated_free": 1, "unknown": 1,
                "spaces": {"B019": "occupied", "B020": "free", "B021": "unknown"}}


class SupabaseDeliveryTests(unittest.TestCase):
    def test_numeric_spot_ids_match_calibration_labels(self):
        self.assertEqual(model_spot_ids(["B001", "B019", "B038"]),
                         {"B001": 1, "B019": 19, "B038": 38})
        with self.assertRaisesRegex(ValueError, "distinct numeric IDs"):
            model_spot_ids(["B019", "B19"])

    @patch.dict("os.environ", {"SUPABASE_URL": "https://example.supabase.co",
                              "SUPABASE_KEY": "sb_secret_test", "SUPABASE_ACCESS_TOKEN": ""})
    @patch("supabase_delivery.HTTPSConnection")
    def test_rejected_post_survives_restart_without_reanalyzing(self, connect):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inbox = root / "inbox"
            inbox.mkdir()
            photo = inbox / "sample.jpg"
            photo.write_bytes(b"test image")
            config = root / "supabase.json"
            config.write_text(json.dumps({
                "table": "parking_spot_updates",
                "parking_lot_id": 1,
            }))
            output = root / "output"
            delivery = SupabaseDelivery(config, output)
            analyzer = TestAnalyzer()
            watcher = FolderWatcher(inbox, output, analyzer, 0, delivery)
            watcher.scan()
            queued, = delivery.pending.glob("*.json")

            connection = connect.return_value
            response = connection.getresponse.return_value
            response.status = 429
            response.read.return_value = b'{"message":"Too many requests"}'
            self.assertFalse(delivery.flush())
            self.assertTrue(queued.exists())
            failed_request = connection.request.call_args.args
            photo.unlink()

            restarted = SupabaseDelivery(config, output)
            response.status = 201
            self.assertTrue(restarted.flush())
            self.assertFalse(queued.exists())
            self.assertTrue((restarted.sent / queued.name).is_file())
            self.assertEqual(analyzer.calls, 1)
            request = connection.request.call_args.args
            self.assertEqual(request, failed_request)
            method, path, body, headers = request
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/rest/v1/parking_spot_updates")
            rows = json.loads(body)
            self.assertEqual(rows, [
                {"parking_lot_id": 1, "parking_spot_id": 19,
                 "parking_spot_type": "standard", "is_available": False},
                {"parking_lot_id": 1, "parking_spot_id": 20,
                 "parking_spot_type": "handicap", "is_available": True},
                {"parking_lot_id": 1, "parking_spot_id": 21,
                 "parking_spot_type": "standard", "is_available": None},
            ])
            self.assertIs(rows[0]["is_available"], False)
            self.assertIs(rows[1]["is_available"], True)
            self.assertEqual(headers["apikey"], "sb_secret_test")
            self.assertIn("missing=default", headers["Prefer"])
            self.assertNotIn("Authorization", headers)
            self.assertTrue(restarted.flush())
            self.assertEqual(connection.request.call_count, 2)

    @patch.dict("os.environ", {"SUPABASE_URL": "https://example.supabase.co",
                              "SUPABASE_KEY": "sb_secret_test"})
    @patch("supabase_delivery.HTTPSConnection")
    def test_post_timeout_is_held_across_restart(self, connect):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "supabase.json"
            config.write_text(json.dumps({
                "table": "parking_spot_updates", "parking_lot_id": 1,
            }))
            delivery = SupabaseDelivery(config, root)
            analyzer = TestAnalyzer()
            watcher = FolderWatcher(root, root, analyzer, 0, delivery)
            (root / "photo.jpg").write_bytes(b"test image")
            watcher.scan()
            connect.return_value.getresponse.side_effect = TimeoutError()
            self.assertFalse(delivery.flush())
            self.assertEqual(len(list(delivery.sending.glob("*.json"))), 1)

            restarted = SupabaseDelivery(config, root)
            FolderWatcher(root, root, analyzer, 0, restarted).scan()
            self.assertFalse(restarted.flush())
            self.assertEqual(analyzer.calls, 1)
            self.assertEqual(connect.return_value.request.call_count, 1)
