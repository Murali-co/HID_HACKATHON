import importlib.util
import json
import math
import os
import sqlite3
import tempfile
import time
import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

warnings.filterwarnings("ignore", message=".*starlette.testclient.*")

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from app import Clock, create_app, decide
from evaluate import summarize
from policy import evaluate
from storage import CorruptTemplateError, Store
from vision import CaptureError, Evidence, Vision, validate_template

SPEC = importlib.util.spec_from_file_location("audit_handler", Path(__file__).parents[1] / "aws/handler.py")
handler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handler)

VALID_TEMPLATE = (b"\x00\x00\x80\x3f" + b"\x00" * 508) # 128-d vector with first elem = 1.0, rest 0.0 (norm=1.0)
BURST = {"subject": "demo-user", "frames": ["A" * 60] * 3}
TOKEN = {"X-Operator-Token": "test-operator-token-123"}


class KioskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        self.vision = Mock()
        self.vision.enroll.return_value = VALID_TEMPLATE
        self.vision.verify.return_value = Evidence("LIVE", .95, .81)
        self.policy = Mock(return_value="authorized")
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"LIVEGUARD_OPERATOR_TOKEN": TOKEN["X-Operator-Token"]}).start()
        app = create_app(lambda **settings: self.vision, lambda: self.store, self.policy)
        self.client = self.enterContext(TestClient(app))

    def scan(self, **overrides):
        response = self.client.post("/api/scan", json={**BURST, **overrides})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_all_required_decisions_and_audit(self):
        self.assertEqual(self.scan()["outcome"], "ALLOW")
        self.vision.verify.return_value = Evidence("SPOOF", .02)
        self.assertEqual(self.scan()["reason"], "presentation_attack")
        self.vision.verify.return_value = Evidence("LIVE", .91, .10)
        self.assertEqual(self.scan()["reason"], "identity_mismatch")
        self.vision.verify.return_value = Evidence("LIVE", .91, .40)
        self.assertEqual(self.scan()["reason"], "identity_uncertain")
        self.assertEqual(len(self.store.pending()), 4)

    def test_retake_rechecks_policy_and_is_single_use(self):
        self.vision.verify.return_value = Evidence("UNCERTAIN", .5, reason="liveness_uncertain")
        initial = self.scan()
        self.assertEqual(initial["outcome"], "REQUEST_VERIFICATION")
        self.vision.verify.return_value = Evidence("LIVE", .95, .90)
        self.policy.return_value = "outside_schedule"
        self.assertEqual(self.scan(retry=initial["retry"])["reason"], "outside_schedule")
        self.assertEqual(self.scan(retry=initial["retry"])["reason"], "retry_expired_or_invalid")

    def test_retake_failure_and_recovery(self):
        self.vision.verify.return_value = Evidence("UNCERTAIN", reason="one_face_required")
        token = self.scan()["retry"]
        self.assertEqual(self.scan(retry=token)["reason"], "verification_failed")
        token = self.scan()["retry"]
        self.vision.verify.return_value = Evidence("LIVE", .92, .75)
        self.assertEqual(self.scan(retry=token)["outcome"], "ALLOW")

    def test_offline_policy_unavailable_and_audit_failure_never_allow(self):
        self.policy.return_value = "policy_unavailable"
        initial = self.scan()
        self.assertEqual(initial["outcome"], "REQUEST_VERIFICATION")
        self.assertIsNotNone(initial.get("retry"))
        self.policy.return_value = "authorized"
        with patch.object(self.store, "audit", side_effect=sqlite3.OperationalError):
            with self.assertLogs("liveguard", "ERROR"):
                result = self.scan()
                self.assertEqual(result["outcome"], "DENY")
                self.assertEqual(result["reason"], "audit_unavailable")
                self.assertIsNone(result.get("retry"), "Retry token must never be leaked on audit denial")

    def test_enrollment_requires_consent_and_operator_and_supports_delete(self):
        self.assertEqual(self.client.post("/api/enroll", json={**BURST, "consent": True}).status_code, 403)
        self.assertEqual(self.client.post("/api/enroll", headers=TOKEN,
                                         json={**BURST, "consent": False}).status_code, 400)
        for expected in (201, 409):
            self.assertEqual(self.client.post("/api/enroll", headers=TOKEN,
                                             json={**BURST, "consent": True}).status_code, expected)
        with self.store.connect() as db:
            sealed = db.execute("SELECT sealed FROM templates").fetchone()[0]
        self.assertNotIn(VALID_TEMPLATE, sealed)
        self.assertEqual(self.store.get("demo-user"), VALID_TEMPLATE)
        self.client.post("/api/delete-enrollment", headers=TOKEN, json={"subject": "demo-user"})
        self.assertIsNone(self.store.get("demo-user"))

    def test_cross_origin_and_invalid_frames_rejected(self):
        response = self.client.post("/api/scan", headers={"Origin": "https://other.example"}, json=BURST)
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/scan", json={**BURST, "frames": ["private-frame"]})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private-frame", response.text)

    def test_unicode_and_kannada_display_name_enrollment(self):
        # Kannada name: "ರಮೇಶ್ ಕುಮಾರ್"
        kannada_name = "ರಮೇಶ್ ಕುಮಾರ್"
        res = self.client.post("/api/enroll", headers=TOKEN,
                               json={**BURST, "subject": "kannada-user", "display_name": kannada_name, "consent": True})
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["display_name"], kannada_name)
        profile = self.store.get_profile("kannada-user")
        self.assertEqual(profile["display_name"], kannada_name)


class SecurityAndBoundaryTests(unittest.TestCase):
    def test_nan_and_infinite_identity_scores_never_allow(self):
        # Critical regression: NaN identity score must NEVER return ALLOW
        ev_nan = Evidence("LIVE", 0.95, float("nan"))
        outcome, reason = decide(ev_nan, "authorized", 0.50, 0.35)
        self.assertEqual(outcome, "REQUEST_VERIFICATION")
        self.assertEqual(reason, "invalid_evidence")

        # Test infinity
        ev_inf = Evidence("LIVE", 0.95, float("inf"))
        outcome, reason = decide(ev_inf, "authorized", 0.50, 0.35)
        self.assertEqual(outcome, "REQUEST_VERIFICATION")
        self.assertEqual(reason, "invalid_evidence")

        # Test out of bounds scores
        ev_oob = Evidence("LIVE", 0.95, 1.5)
        outcome, reason = decide(ev_oob, "authorized", 0.50, 0.35)
        self.assertEqual(outcome, "REQUEST_VERIFICATION")
        self.assertEqual(reason, "invalid_evidence")

    def test_template_validation_rejects_corrupted_vectors(self):
        # Non-finite values
        nan_vec = (float("nan") * np.ones(128, dtype=np.float32)).tobytes()
        with self.assertRaises(ValueError):
            validate_template(nan_vec)

        # Truncated vector
        short_vec = b"\x00" * 200
        with self.assertRaises(ValueError):
            validate_template(short_vec)

        # Unnormalized vector
        unnorm_vec = (np.ones(128, dtype=np.float32) * 5.0).tobytes()
        with self.assertRaises(ValueError):
            validate_template(unnorm_vec)

        # Valid unit vector
        valid = np.zeros(128, dtype=np.float32)
        valid[0] = 1.0
        self.assertEqual(len(validate_template(valid.tobytes())), 128)

    def test_store_corrupt_template_raises_error(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp))
            with store.connect() as db:
                db.execute("INSERT INTO templates (subject, sealed) VALUES (?, ?)",
                           ("bad-user", b"garbage_nonce_and_ciphertext"))
            with self.assertRaises(CorruptTemplateError):
                store.get("bad-user")

    def test_additive_migration_preserves_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            db_path = state_dir / "kiosk.sqlite"
            # Create old 2-column schema
            con = sqlite3.connect(db_path)
            con.execute("CREATE TABLE templates (subject TEXT PRIMARY KEY, sealed BLOB NOT NULL)")
            con.execute("INSERT INTO templates VALUES ('legacy-user', X'01020304')")
            con.commit()
            con.close()

            # Initialize modern Store on existing file
            store = Store(state_dir)
            with store.connect() as db:
                cols = {row[1] for row in db.execute("PRAGMA table_info(templates)").fetchall()}
                self.assertIn("display_name", cols)
                self.assertIn("created_at", cols)
                self.assertIn("enrolled_by", cols)
                row = db.execute("SELECT subject, sealed, display_name FROM templates WHERE subject='legacy-user'").fetchone()
                self.assertEqual(row[0], "legacy-user")
                self.assertIsNone(row[2])


class SupportingTests(unittest.TestCase):
    def test_signed_policy_accepts_and_rejects_tamper_expiry(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from policy import encoded
        import base64

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            private = Ed25519PrivateKey.generate()
            (directory / "policy.pub").write_bytes(private.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
            policy = {"issued": 100, "expires": 200, "timezone": "UTC", "door": "campus-lab",
                      "access": {"demo-user": [0, 1440]}}
            signed = {"policy": policy, "signature": base64.b64encode(private.sign(encoded(policy))).decode()}
            file = directory / "policy.json"
            file.write_bytes(encoded(signed))
            self.assertEqual(evaluate("demo-user", 150, directory), "authorized")
            self.assertEqual(evaluate("other", 150, directory), "outside_schedule")
            self.assertEqual(evaluate("demo-user", 200, directory), "policy_unavailable")
            policy["expires"] = 300
            file.write_bytes(encoded(signed))
            self.assertEqual(evaluate("demo-user", 150, directory), "policy_unavailable")

    def test_clock_discontinuity_fails_closed(self):
        clock = Clock()
        with patch("app.time.time", return_value=clock.wall - 120):
            self.assertIsNone(clock.read())
        self.assertIsNone(clock.read())

    def test_ingest_idempotency_and_data_minimization(self):
        body = {"event_id": str(uuid.uuid4()), "created": int(time.time()), "outcome": "ALLOW"}
        with patch.dict(os.environ, {"TABLE": "local-test"}), patch.object(handler.boto3, "resource") as dynamo:
            table = dynamo.return_value.Table.return_value
            table.put_item.side_effect = [None, ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")]
            for _ in range(2):
                self.assertEqual(handler.ingest({"body": json.dumps(body)}, None)["statusCode"], 204)
            self.assertEqual(table.put_item.call_count, 2)
            self.assertEqual(set(table.put_item.call_args.kwargs["Item"]), set(body) | {"expires"})
            self.assertEqual(handler.ingest({"body": json.dumps({**body, "frame": "abc"})}, None)["statusCode"], 400)
            self.assertEqual(table.put_item.call_count, 2)

    def test_aws_handler_base64_decoding(self):
        import base64
        body = {"event_id": str(uuid.uuid4()), "created": int(time.time()), "outcome": "ALLOW"}
        raw_json = json.dumps(body)
        b64_body = base64.b64encode(raw_json.encode()).decode()
        event = {"body": b64_body, "isBase64Encoded": True}
        with patch.dict(os.environ, {"TABLE": "local-test"}), patch.object(handler.boto3, "resource") as dynamo:
            table = dynamo.return_value.Table.return_value
            table.put_item.return_value = {}
            res = handler.ingest(event, None)
            self.assertEqual(res["statusCode"], 204)

    def test_evaluation_counts_uncertain_and_deadline_misses(self):
        report = summarize([
            {"scenario": "print", "attempt": "initial", "pad": "UNCERTAIN",
             "outcome": "REQUEST_VERIFICATION", "browser_ms": "2001"},
            {"scenario": "print", "attempt": "initial", "pad": "SPOOF",
             "outcome": "DENY", "browser_ms": "442"},
            {"scenario": "print", "attempt": "retake", "pad": "SPOOF",
             "outcome": "DENY", "browser_ms": "501"},
        ])["print"]
        self.assertEqual(report["attack_pad_detection_rate"], .5)
        self.assertEqual(report["pad_uncertain_rate"], .5)
        self.assertEqual(report["within_2s"], 1)
        self.assertEqual(report["browser_p95_ms"], 2001)

    def test_evaluation_robustness_with_floats_and_empty_values(self):
        report = summarize([
            {"scenario": "genuine", "attempt": "initial", "pad": "LIVE",
             "outcome": "ALLOW", "browser_ms": "123.4", "server_ms": "50.2"},
            {"scenario": "genuine", "attempt": "initial", "pad": "LIVE",
             "outcome": "ALLOW", "browser_ms": "180.9", "server_ms": "None"},
            {"scenario": "genuine", "attempt": "initial", "pad": "LIVE",
             "outcome": "ALLOW", "browser_ms": "NaN", "server_ms": ""},
        ])["genuine"]
        self.assertEqual(report["within_2s"], 2)
        self.assertEqual(report["browser_max_ms"], 180.9)
        self.assertAlmostEqual(report["server_p95_ms"], 50.2)

    def test_decode_frame_size_and_error_handling(self):
        import base64
        import io
        from PIL import Image
        from vision import decode, CaptureError, crop_with_context

        # Frame too small: 100x100
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="JPEG")
        b64_small = base64.b64encode(buf.getvalue()).decode()
        with self.assertRaises(CaptureError) as ctx_err:
            decode(b64_small)
        self.assertEqual(str(ctx_err.exception), "invalid_frame_size")

        # Invalid non-JPEG data
        with self.assertRaises(CaptureError) as ctx_err:
            decode(base64.b64encode(b"not an image at all").decode())
        self.assertEqual(str(ctx_err.exception), "invalid_frame")

        # Valid 320x240 frame
        buf = io.BytesIO()
        Image.new("RGB", (320, 240), color=(100, 150, 200)).save(buf, format="JPEG")
        b64_valid = base64.b64encode(buf.getvalue()).decode()
        decoded = decode(b64_valid)
        self.assertEqual(decoded.shape, (240, 320, 3))

        # crop_with_context zero/negative bounds check
        with self.assertRaises(CaptureError) as ctx_err:
            crop_with_context(decoded, [0, 0, 0, 0], 2.7)
        self.assertEqual(str(ctx_err.exception), "invalid_face_crop")

    def test_real_vision_inference_pipeline_and_enroll_pad(self):
        import base64
        import io
        import cv2
        import numpy as np
        from PIL import Image
        from vision import Vision

        # Synthesize a non-live drawn face
        img = np.full((240, 320, 3), 130, dtype=np.uint8)
        cv2.ellipse(img, (160, 120), (50, 65), 0, 0, 360, (210, 210, 210), -1)
        cv2.circle(img, (140, 105), 6, (40, 40, 40), -1)
        cv2.circle(img, (180, 105), 6, (40, 40, 40), -1)
        cv2.line(img, (160, 115), (160, 130), (70, 70, 70), 2)
        cv2.ellipse(img, (160, 145), (16, 6), 0, 0, 180, (40, 40, 40), 2)

        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
        frame = base64.b64encode(buf.getvalue()).decode()

        # Enrolling synthetic drawn face with check_enroll_pad=True MUST reject as non-live
        vision_strict = Vision(check_enroll_pad=True)
        with self.assertRaises(CaptureError) as exc_pad:
            vision_strict.enroll([frame, frame, frame])
        self.assertIn(str(exc_pad.exception), ("enrollment_liveness_uncertain", "enrollment_presentation_attack"))

        # When check_enroll_pad=False (smoke-test mode), template is generated
        vision_smoke = Vision(check_enroll_pad=False)
        template = vision_smoke.enroll([frame, frame, frame])
        self.assertEqual(len(template), 512)

        ev = vision_smoke.verify([frame, frame, frame], template)
        self.assertIn(ev.pad, ("LIVE", "UNCERTAIN", "SPOOF"))
        self.assertIsInstance(ev.live_score, float)
        self.assertIn("frames", ev.diagnostics)


class FullWorkflowIntegrationTests(unittest.TestCase):
    def test_end_to_end_enroll_scan_retake_and_deletion(self):
        import base64
        import io
        import cv2
        import numpy as np
        from PIL import Image
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from policy import encoded, install
        from vision import Vision

        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            priv = Ed25519PrivateKey.generate()
            now = int(time.time())
            pol = {"issued": now - 10, "expires": now + 3600, "door": "campus-lab", "timezone": "UTC",
                   "access": {"demo-user": [0, 1440]}}
            env = {"policy": pol, "signature": base64.b64encode(priv.sign(encoded(pol))).decode()}
            install(state_dir / "policy.pub", priv.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
            install(state_dir / "policy.json", encoded(env))

            img = np.full((240, 320, 3), 130, dtype=np.uint8)
            cv2.ellipse(img, (160, 120), (50, 65), 0, 0, 360, (210, 210, 210), -1)
            cv2.circle(img, (140, 105), 6, (40, 40, 40), -1)
            cv2.circle(img, (180, 105), 6, (40, 40, 40), -1)
            cv2.line(img, (160, 115), (160, 130), (70, 70, 70), 2)
            cv2.ellipse(img, (160, 145), (16, 6), 0, 0, 180, (40, 40, 40), 2)

            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
            frame = base64.b64encode(buf.getvalue()).decode()

            store = Store(state_dir)
            # Use check_enroll_pad=False for synthetic drawing simulation
            vision = Vision(check_enroll_pad=False)
            app = create_app(lambda **settings: vision, lambda: store, lambda sub, t: evaluate(sub, t, state_dir))

            with patch.dict(os.environ, {"LIVEGUARD_OPERATOR_TOKEN": "test-operator-token-456"}):
                with TestClient(app) as client:
                    # 1. Status endpoint
                    res_status = client.get("/api/status")
                    self.assertEqual(res_status.status_code, 200)
                    self.assertTrue(res_status.json()["models_ready"])

                    # 2. Non-enrolled user scan
                    res_scan_unregistered = client.post("/api/scan", json={"subject": "unknown-user", "frames": [frame]*3})
                    self.assertEqual(res_scan_unregistered.status_code, 200)
                    self.assertIn(res_scan_unregistered.json()["outcome"], ("DENY", "REQUEST_VERIFICATION"))

                    # 3. Enroll demo-user with display name
                    headers = {"X-Operator-Token": "test-operator-token-456", "X-Operator-Id": "admin-1"}
                    res_enroll = client.post("/api/enroll",
                                             json={"subject": "demo-user", "display_name": "Demo Admin", "frames": [frame]*3, "consent": True},
                                             headers=headers)
                    self.assertEqual(res_enroll.status_code, 201)
                    self.assertTrue(res_enroll.json()["enrolled"])
                    self.assertEqual(res_enroll.json()["display_name"], "Demo Admin")

                    # 4. Scan enrolled user
                    res_scan = client.post("/api/scan", json={"subject": "demo-user", "frames": [frame]*3})
                    self.assertEqual(res_scan.status_code, 200)
                    scan_data = res_scan.json()
                    self.assertIn("server_ms", scan_data)
                    self.assertIn("diagnostics", scan_data)
                    self.assertEqual(scan_data["display_name"], "Demo Admin")
                    self.assertLess(scan_data["server_ms"], 1250)

                    # 5. Retry token consumption
                    if scan_data["outcome"] == "REQUEST_VERIFICATION" and scan_data["retry"]:
                        token = scan_data["retry"]
                        res_retry = client.post("/api/scan", json={"subject": "demo-user", "frames": [frame]*3, "retry": token})
                        self.assertEqual(res_retry.status_code, 200)
                        self.assertEqual(res_retry.json()["outcome"], "DENY")
                        self.assertEqual(res_retry.json()["reason"], "verification_failed")

                        # Test token cannot be reused
                        res_reuse = client.post("/api/scan", json={"subject": "demo-user", "frames": [frame]*3, "retry": token})
                        self.assertEqual(res_reuse.json()["reason"], "retry_expired_or_invalid")

                    # 6. Delete enrollment
                    res_delete = client.post("/api/delete-enrollment", json={"subject": "demo-user"}, headers=headers)
                    self.assertEqual(res_delete.status_code, 200)
                    self.assertTrue(res_delete.json()["deleted"])
                    self.assertIsNone(store.get("demo-user"))


if __name__ == "__main__":
    unittest.main()
