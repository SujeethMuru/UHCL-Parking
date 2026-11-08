"""Analyze completed 360 JPEGs and optionally post the results to Supabase."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import time

import cv2
import numpy as np

from local_storage import write_artifact
from multi_view import infer
from occupancy_state import global_observations
from parking_map import combined_counts, validate_map
from video_occupancy import make_projector
from view_gallery import gallery_views


class SnapshotAnalyzer:
    def __init__(self, config, weights):
        from ultralytics import YOLO

        calibration = config.read_bytes()
        self.calibration_id = hashlib.sha256(calibration).hexdigest()
        self.mapping = json.loads(calibration)
        validate_map(self.mapping)
        if not self.mapping["spaces"]:
            raise ValueError("Choose a calibration with marked parking spaces.")
        view_ids = {view["id"] for view in gallery_views(self.mapping)}
        view_ids.update(space["owner"] for space in self.mapping["spaces"])
        self.views = [view for view in self.mapping["views"] if view["id"] in view_ids]
        self.model = YOLO(str(weights))
        self.image_size = None
        self.projectors = {}

    def analyze(self, jpeg):
        if not jpeg.startswith(b"\xff\xd8") or jpeg.rfind(b"\xff\xd9") < 0:
            raise ValueError("JPEG is incomplete or unreadable.")
        panorama = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if panorama is None:
            raise ValueError("JPEG could not be decoded.")
        height, width = panorama.shape[:2]
        if abs(width / height - 2) > 0.05:
            raise ValueError("Expected a stitched 2:1 equirectangular JPEG.")
        if self.image_size != (width, height):
            self.projectors = {
                view["id"]: make_projector(width, height, view["yaw"], view["pitch"], view["fov"])
                for view in self.views
            }
            self.image_size = (width, height)
        detections = {
            key: infer(self.model, project(panorama))
            for key, project in self.projectors.items()
        }
        observations, _ = global_observations(self.mapping, detections)
        return combined_counts(self.mapping, observations)


def file_version(stat):
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class FolderWatcher:
    def __init__(self, inbox, output, analyzer, settle_seconds, delivery=None):
        self.inbox = inbox
        self.output = delivery.pending if delivery else output
        self.uploaded = (delivery.sending, delivery.sent) if delivery else ()
        self.analyzer = analyzer
        self.space_types = {space["id"]: space.get("space_type", "standard")
                            for space in analyzer.mapping["spaces"]}
        self.settle_seconds = settle_seconds
        self.pending = {}

    def scan(self):
        # Scan existing files too, so captures made while the laptop was away
        # can be imported. Process older files first when several arrive together.
        files = []
        for path in self.inbox.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            try:
                if path.is_file():
                    files.append((path, file_version(path.stat())))
            except OSError as exc:
                logging.warning("Could not inspect %s: %s", path.name, exc)
        now = time.monotonic()
        previous = self.pending
        self.pending = {}
        failures = 0
        for path, version in sorted(files, key=lambda item: (item[1][1], item[0].name)):
            identity = json.dumps([self.analyzer.calibration_id, str(path), version])
            snapshot_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            result = self.output / f"{snapshot_id}.json"
            if result.exists() or any((folder / result.name).exists() for folder in self.uploaded):
                continue
            ready_at = previous.get(snapshot_id, now + self.settle_seconds)
            if now < ready_at:
                self.pending[snapshot_id] = ready_at
                continue
            try:
                jpeg = path.read_bytes()
                if len(jpeg) != version[0] or file_version(path.stat()) != version:
                    continue
                counts = self.analyzer.analyze(jpeg)
                if file_version(path.stat()) != version:
                    continue
                payload = {
                    "schema_version": 1,
                    "snapshot_id": snapshot_id,
                    "lot": self.analyzer.mapping["lot"],
                    "calibration_id": self.analyzer.calibration_id,
                    "source_file": path.name,
                    "file_modified_at": datetime.fromtimestamp(version[1] / 1e9, timezone.utc).isoformat(),
                    "analyzed_at": datetime.now(timezone.utc).isoformat(),
                    "counts": counts,
                    "space_types": self.space_types,
                }
                saved = write_artifact(result, json.dumps(payload, indent=2).encode("utf-8"))
                logging.info("%s: %s occupied, %s estimated free, %s unknown -> %s",
                             path.name, counts["occupied"], counts["estimated_free"],
                             counts["unknown"], saved)
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                failures += 1
                logging.warning("Could not process %s (%s). Retrying in 60 seconds.", path.name, exc)
                self.pending[snapshot_id] = time.monotonic() + 60
        return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Map aligned to the camera's fixed pose.")
    parser.add_argument("--output", type=Path, default=Path("outputs/snapshots"))
    parser.add_argument("--supabase-config", type=Path, help="Enable Supabase delivery using this table mapping.")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--settle-seconds", type=float, default=4)
    parser.add_argument("--once", action="store_true", help="Process the current batch and exit.")
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.poll_seconds, args.settle_seconds)):
        parser.error("Polling and settling times must be finite and positive.")
    weights = Path("models/yolo11n.pt")
    if not weights.is_file():
        parser.error("Missing models/yolo11n.pt. See the parking-cv setup instructions.")
    output = args.output.resolve()
    delivery = None
    if args.supabase_config:
        from supabase_delivery import SupabaseDelivery

        try:
            delivery = SupabaseDelivery(args.supabase_config, output)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            parser.error(f"Could not configure Supabase: {exc}")
    try:
        analyzer = SnapshotAnalyzer(args.config, weights)
        if delivery:
            delivery.validate_calibration(analyzer.mapping)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Could not load the calibration: {exc}")
    watcher = FolderWatcher(args.inbox.resolve(), output, analyzer, args.settle_seconds, delivery)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Watching %s for stitched 360 JPEGs.", watcher.inbox)
    if delivery and any(delivery.sending.glob("*.json")):
        logging.warning("Review unconfirmed Supabase uploads in %s.", delivery.sending)
    try:
        if args.once:
            failed = False
            try:
                watcher.scan()
                time.sleep(args.settle_seconds)
                failed = watcher.scan() > 0
            except OSError as exc:
                logging.error("Could not scan %s: %s", watcher.inbox, exc)
                failed = True
            delivered = delivery is None or delivery.flush()
            return int(failed or not delivered)
        while True:
            try:
                watcher.scan()
            except OSError as exc:
                logging.warning("Could not scan %s: %s", watcher.inbox, exc)
            if delivery:
                delivery.flush()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        logging.info("Stopped watching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
