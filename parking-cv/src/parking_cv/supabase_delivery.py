"""Deliver queued parking reports to an existing Supabase table."""

import hashlib
from http.client import HTTPException, HTTPSConnection
import json
import logging
import os
import re
import time
from urllib.parse import quote, urlsplit


def model_spot_ids(labels):
    ids = {}
    for label in labels:
        if not re.fullmatch(r"B[0-9]+", label) or int(label[1:]) < 1:
            raise ValueError(f"Expected a calibrated spot label like B019, got {label!r}.")
        ids[label] = int(label[1:])
    if len(ids) != len(set(ids.values())):
        raise ValueError("Calibrated spot labels must have distinct numeric IDs.")
    return ids


class SupabaseDelivery:
    def __init__(self, config, output):
        settings = json.loads(config.read_text(encoding="utf-8-sig"))
        url = urlsplit(os.environ.get("SUPABASE_URL", ""))
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.path not in {"", "/"} or url.query or url.fragment):
            raise ValueError("Set SUPABASE_URL to the HTTPS project URL.")
        key = os.environ.get("SUPABASE_KEY", "").strip()
        if not key:
            raise ValueError("Set SUPABASE_KEY in the watcher's environment.")

        schema = settings.get("schema", "public")
        table = settings["table"]
        self.lot_id = settings["parking_lot_id"]
        names = [schema, table]
        if not all(isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                   for name in names):
            raise ValueError("Use plain SQL identifiers for the schema and table.")
        if type(self.lot_id) is not int or self.lot_id < 0:
            raise ValueError("parking_lot_id must be a nonnegative integer.")

        self.host = url.hostname
        self.port = url.port
        self.path = f"/rest/v1/{quote(table)}"
        self.headers = {
            "apikey": key,
            "Content-Type": "application/json",
            "Content-Profile": schema,
            "Prefer": "return=minimal,missing=default",
        }
        token = os.environ.get("SUPABASE_ACCESS_TOKEN") or (key if key.startswith("eyJ") else None)
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        destination = json.dumps([url.netloc, schema, table, self.lot_id])
        queue_id = hashlib.sha256(destination.encode("utf-8")).hexdigest()[:16]
        self.pending = output / f"pending-{queue_id}"
        self.sending = output / f"sending-{queue_id}"
        self.sent = output / f"sent-{queue_id}"
        self.retry_at = 0

    def validate_calibration(self, mapping):
        model_spot_ids(space["id"] for space in mapping["spaces"])

    def rows(self, report):
        states = report["counts"]["spaces"]
        spot_ids = model_spot_ids(states)
        types = {"standard": "standard", "accessible": "handicap",
                 "ev": "ev", "accessible_ev": "accessible_ev"}
        availability = {"free": True, "occupied": False, "unknown": None}
        return [
            {"parking_lot_id": self.lot_id, "parking_spot_id": spot_id,
             "parking_spot_type": types[report["space_types"][label]],
             "is_available": availability[states[label]]}
            for label, spot_id in sorted(spot_ids.items(), key=lambda item: item[1])
        ]

    def post(self, path, rows):
        body = json.dumps(rows).encode("utf-8")
        in_flight = self.sending / path.name
        self.sending.mkdir(parents=True, exist_ok=True)
        connection = HTTPSConnection(self.host, self.port, timeout=10)
        try:
            connection.connect()
            # Once sending starts, a timeout or interruption may hide a committed insert.
            path.replace(in_flight)
            connection.request("POST", self.path, body, self.headers)
            response = connection.getresponse()
            if 200 <= response.status < 300:
                in_flight.replace(self.sent / path.name)
                return
            if 400 <= response.status < 500 and response.status != 408:
                in_flight.replace(path)
            try:
                error = json.loads(response.read(4096))
                message = error.get("message", "")
            except (ValueError, AttributeError):
                message = ""
            status = f"Supabase returned HTTP {response.status}"
            raise RuntimeError(f"{status}: {message}" if message else status)
        finally:
            connection.close()

    def flush(self):
        queued = sorted(self.pending.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
        if not queued:
            return not any(self.sending.glob("*.json"))
        if time.monotonic() < self.retry_at:
            return False
        for path in queued:
            try:
                report = json.loads(path.read_bytes())
                rows = self.rows(report)
                if not rows:
                    raise ValueError("Snapshot has no calibrated parking spaces.")
                self.sent.mkdir(parents=True, exist_ok=True)
                self.post(path, rows)
                logging.info("Posted %s parking spaces to Supabase for %s.", len(rows), path.stem)
            except (OSError, HTTPException, RuntimeError, ValueError, KeyError, TypeError) as exc:
                if (self.sending / path.name).exists():
                    logging.error("Upload outcome unknown for %s (%s). Held in %s for review.",
                                  path.stem, exc, self.sending)
                    continue
                logging.warning("Supabase upload pending for %s (%s). Retrying in 60 seconds.", path.stem, exc)
                self.retry_at = time.monotonic() + 60
                return False
        return not any(self.sending.glob("*.json"))
