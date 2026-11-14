"""LiveGuard loopback kiosk API. Run with one Uvicorn worker."""
import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from policy import evaluate
from storage import CorruptTemplateError, Store
from vision import CaptureError, Evidence, Vision

WEB = Path(__file__).resolve().parent / "static"
Subject = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,32}$")]
DisplayName = Annotated[str, StringConstraints(min_length=1, max_length=64)]
JPEG = Annotated[str, StringConstraints(min_length=40, max_length=180_000)]
LIMIT = 560_000
LOGGER = logging.getLogger("liveguard")


class SubjectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: Subject


class Burst(SubjectBody):
    frames: Annotated[list[JPEG], Field(min_length=3, max_length=3)]


class Enroll(Burst):
    display_name: DisplayName | None = None
    consent: bool


class Scan(Burst):
    retry: Annotated[str, StringConstraints(max_length=48)] | None = None


class RetryBook:
    def __init__(self):
        self.items = {}
        self.lock = threading.Lock()

    def issue(self, subject):
        token = secrets.token_urlsafe(24)
        now = time.monotonic()
        with self.lock:
            self.items = {key: value for key, value in self.items.items() if value[1] > now}
            if len(self.items) >= 64:
                return None
            self.items[token] = (subject, now + 30)
        return token

    def consume(self, token, subject):
        with self.lock:
            saved = self.items.pop(token, None)
        return saved is not None and saved[0] == subject and saved[1] > time.monotonic()

    def invalidate(self, token):
        with self.lock:
            self.items.pop(token, None)


class Clock:
    """Process-local clock discontinuity check for a supervised demo."""
    def __init__(self):
        self.wall, self.monotonic = time.time(), time.monotonic()
        self.valid = True
        self.lock = threading.Lock()

    def read(self):
        now = time.time()
        with self.lock:
            elapsed = time.monotonic() - self.monotonic
            self.valid = self.valid and abs(now - self.wall - elapsed) <= 5
            return now if self.valid else None


def decide(evidence, permission, match_min, mismatch_max):
    # Strict validation of evidence structure and finite numeric bounds
    if not evidence or evidence.pad not in ("LIVE", "SPOOF", "UNCERTAIN"):
        return "REQUEST_VERIFICATION", "invalid_evidence"

    if evidence.live_score is not None:
        if not (isinstance(evidence.live_score, (int, float)) and math.isfinite(evidence.live_score) and 0.0 <= evidence.live_score <= 1.0):
            return "REQUEST_VERIFICATION", "invalid_evidence"

    if evidence.identity_score is not None:
        if not (isinstance(evidence.identity_score, (int, float)) and math.isfinite(evidence.identity_score) and -1.0 <= evidence.identity_score <= 1.0):
            return "REQUEST_VERIFICATION", "invalid_evidence"

    if evidence.pad == "SPOOF":
        return "DENY", "presentation_attack"

    if permission == "outside_schedule":
        return "DENY", "outside_schedule"

    if evidence.pad == "LIVE":
        if evidence.identity_score is None:
            return "DENY", "identity_not_enrolled"
        if evidence.identity_score < mismatch_max:
            return "DENY", "identity_mismatch"

    if permission != "authorized":
        return "REQUEST_VERIFICATION", "policy_unavailable"

    if evidence.pad != "LIVE":
        return "REQUEST_VERIFICATION", evidence.reason or "liveness_uncertain"

    if evidence.identity_score < match_min:
        return "REQUEST_VERIFICATION", "identity_uncertain"

    return "ALLOW", "verified"


def verify_operator(operator_id: str | None, token: str | None):
    # Support multiple operators via LIVEGUARD_OPERATORS (JSON: {"op_id": "token_str", ...})
    multi_ops = os.environ.get("LIVEGUARD_OPERATORS")
    if multi_ops:
        try:
            allowed = json.loads(multi_ops)
            if isinstance(allowed, dict) and operator_id in allowed:
                expected = allowed[operator_id]
                if len(expected) >= 16 and secrets.compare_digest((token or "").encode(), expected.encode()):
                    return operator_id
        except Exception:
            pass

    expected = os.environ.get("LIVEGUARD_OPERATOR_TOKEN", "")
    expected_id = os.environ.get("LIVEGUARD_OPERATOR_ID")
    if expected_id and operator_id and operator_id != expected_id:
        raise HTTPException(403, "operator_required")
    if len(expected) < 16 or not secrets.compare_digest((token or "").encode(), expected.encode()):
        raise HTTPException(403, "operator_required")
    return operator_id or expected_id or "operator"


def create_app(vision_factory=Vision, store_factory=Store, check_policy=evaluate, clock_factory=Clock):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store_factory()
        app.state.clock = clock_factory()
        app.state.retry = RetryBook()
        app.state.match_min = float(os.environ.get("LIVEGUARD_MATCH_MIN", ".50"))
        app.state.mismatch_max = float(os.environ.get("LIVEGUARD_MISMATCH_MAX", ".35"))
        if not -1 < app.state.mismatch_max < app.state.match_min < 1:
            raise ValueError("Invalid identity threshold band")
        try:
            app.state.vision = vision_factory(live_min=float(os.environ.get("LIVEGUARD_LIVE_MIN", ".85")),
                                             spoof_max=float(os.environ.get("LIVEGUARD_SPOOF_MAX", ".25")))
        except (RuntimeError, ValueError):
            LOGGER.exception("Model setup failed")
            app.state.vision = None
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.vision = None
    app.state.store = None
    app.state.clock = None
    app.state.retry = None
    app.state.match_min = 0.50
    app.state.mismatch_max = 0.35
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        request.state.start = time.perf_counter()
        if request.method == "POST":
            origin = request.headers.get("origin")
            local_origin = f"{request.url.scheme}://{request.url.netloc}"
            if origin is not None and origin != local_origin:
                return JSONResponse({"detail": "cross_origin_request"}, status_code=403)
            size = request.headers.get("content-length", "")
            if not size.isdecimal() or not 0 < int(size) <= LIMIT or "transfer-encoding" in request.headers:
                return JSONResponse({"detail": "invalid_body_size"}, status_code=413)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "json_required"}, status_code=415)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; object-src 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_capture(request, error):
        return JSONResponse({"detail": "invalid_capture_request"}, status_code=422)

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/status")
    def status():
        return {"models_ready": app.state.vision is not None, "door": "campus-lab"}

    @app.post("/api/enroll", status_code=201)
    def enroll(body: Enroll,
               x_operator_token: Annotated[str | None, Header()] = None,
               x_operator_id: Annotated[str | None, Header()] = None):
        op_id = verify_operator(x_operator_id, x_operator_token)
        if not body.consent:
            raise HTTPException(400, "consent_required")
        if app.state.vision is None:
            raise HTTPException(503, "models_unavailable")

        clean_display_name = body.display_name.strip() if body.display_name else None
        if body.display_name is not None and not clean_display_name:
            raise HTTPException(422, "invalid_display_name")

        try:
            template = app.state.vision.enroll(body.frames)
            app.state.store.enroll(body.subject, template, display_name=clean_display_name, enrolled_by=op_id)
        except CaptureError as exc:
            raise HTTPException(422, str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, "identity_already_enrolled") from exc
        return {"subject": body.subject, "display_name": clean_display_name, "enrolled": True}

    @app.post("/api/delete-enrollment")
    def delete_enrollment(body: SubjectBody,
                          x_operator_token: Annotated[str | None, Header()] = None,
                          x_operator_id: Annotated[str | None, Header()] = None):
        verify_operator(x_operator_id, x_operator_token)
        return {"subject": body.subject, "deleted": app.state.store.delete(body.subject)}

    @app.post("/api/scan")
    def scan(body: Scan, request: Request):
        start = request.state.start
        evidence = None
        retry_token = None
        outcome, reason = "DENY", "retry_expired_or_invalid"

        if body.retry is None or app.state.retry.consume(body.retry, body.subject):
            try:
                if app.state.vision is None:
                    outcome, reason = "REQUEST_VERIFICATION", "models_unavailable"
                else:
                    template = None
                    try:
                        template = app.state.store.get(body.subject)
                    except CorruptTemplateError:
                        LOGGER.exception("Corrupt template for subject %s", body.subject)
                        outcome, reason = "REQUEST_VERIFICATION", "template_corrupt"
                        evidence = Evidence("UNCERTAIN", reason="template_corrupt")

                    if evidence is None:
                        evidence = app.state.vision.verify(body.frames, template)
                        now = app.state.clock.read()
                        permission = check_policy(body.subject, now) if now is not None else "policy_unavailable"
                        outcome, reason = decide(evidence, permission, app.state.match_min, app.state.mismatch_max)
            except Exception:
                LOGGER.exception("Scan failed")
                outcome, reason = "REQUEST_VERIFICATION", "pipeline_unavailable"

            if outcome != "DENY" and time.perf_counter() - start > 1.25:
                outcome, reason = "REQUEST_VERIFICATION", "decision_timeout"

            if outcome == "REQUEST_VERIFICATION":
                if body.retry is not None:
                    outcome, reason = "DENY", "verification_failed"
                else:
                    retry_token = app.state.retry.issue(body.subject)

        event_id = str(uuid.uuid4())
        try:
            app.state.store.audit(event_id, int(time.time()), outcome)
        except sqlite3.Error:
            LOGGER.exception("Audit unavailable")
            outcome, reason = "DENY", "audit_unavailable"
            if retry_token:
                app.state.retry.invalidate(retry_token)
                retry_token = None

        profile = None
        try:
            profile = app.state.store.get_profile(body.subject)
        except Exception:
            pass
        display_name = profile["display_name"] if profile else None

        return {
            "event_id": event_id,
            "outcome": outcome,
            "reason": reason,
            "subject": body.subject,
            "display_name": display_name,
            "retry": retry_token,
            "pad": evidence.pad if evidence else "UNCERTAIN",
            "live_score": round(evidence.live_score, 4) if evidence and evidence.live_score is not None else None,
            "identity_score": round(evidence.identity_score, 4)
            if evidence and evidence.identity_score is not None else None,
            "server_ms": round((time.perf_counter() - start) * 1000, 1),
            "diagnostics": evidence.diagnostics if evidence else {},
        }

    return app


app = create_app()
