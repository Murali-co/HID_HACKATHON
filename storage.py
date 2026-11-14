"""SQLite templates and audit queue. Images are held only in memory."""
import os
import sqlite3
import time
from contextlib import contextmanager

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from policy import STATE


class CorruptTemplateError(Exception):
    """Raised when an enrolled template cannot be decrypted or has invalid structure."""
    pass


class Store:
    def __init__(self, state=STATE):
        state.mkdir(exist_ok=True, parents=True, mode=0o700)
        self.path = state / "kiosk.sqlite"
        key_path = state / "templates.key"
        try:
            with os.fdopen(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as key_file:
                key_file.write(AESGCM.generate_key(256))
        except FileExistsError:
            pass
        self.cipher = AESGCM(key_path.read_bytes())
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS templates (
                subject TEXT PRIMARY KEY,
                sealed BLOB NOT NULL,
                display_name TEXT,
                created_at INTEGER,
                enrolled_by TEXT
            )""")
            # Additive migration: safely add columns if opening an existing legacy database
            existing_cols = {row[1] for row in db.execute("PRAGMA table_info(templates)").fetchall()}
            if "display_name" not in existing_cols:
                db.execute("ALTER TABLE templates ADD COLUMN display_name TEXT")
            if "created_at" not in existing_cols:
                db.execute("ALTER TABLE templates ADD COLUMN created_at INTEGER")
            if "enrolled_by" not in existing_cols:
                db.execute("ALTER TABLE templates ADD COLUMN enrolled_by TEXT")

            db.execute("""CREATE TABLE IF NOT EXISTS audit (
                event_id TEXT PRIMARY KEY,
                created INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                synced INTEGER NOT NULL DEFAULT 0
            )""")

    @contextmanager
    def connect(self):
        # 0.8s bounded timeout: allows concurrent WAL access while preserving backend 1.25s budget
        db = sqlite3.connect(self.path, timeout=0.8)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, subject):
        """Retrieve and decrypt the biometric template for a subject.
        Returns bytes if enrolled and valid, None if not enrolled.
        Raises CorruptTemplateError if enrolled but corrupted/tampered.
        """
        with self.connect() as db:
            row = db.execute("SELECT sealed FROM templates WHERE subject=?", (subject,)).fetchone()
        if not row:
            return None
        sealed = row[0]
        try:
            if not isinstance(sealed, (bytes, bytearray)) or len(sealed) < 28:
                raise CorruptTemplateError("truncated_sealed_data")
            raw = self.cipher.decrypt(sealed[:12], sealed[12:], subject.encode())
            if len(raw) != 512:
                raise CorruptTemplateError("invalid_template_length")
            return raw
        except CorruptTemplateError:
            raise
        except Exception as exc:
            raise CorruptTemplateError("decryption_failed") from exc

    def get_profile(self, subject):
        """Retrieve subject metadata if enrolled."""
        with self.connect() as db:
            row = db.execute(
                "SELECT subject, display_name, created_at, enrolled_by FROM templates WHERE subject=?",
                (subject,)
            ).fetchone()
        if row:
            return {
                "subject": row[0],
                "display_name": row[1],
                "created_at": row[2],
                "enrolled_by": row[3],
            }
        return None

    def enroll(self, subject, template, display_name=None, enrolled_by=None):
        if not isinstance(template, (bytes, bytearray)) or len(template) != 512:
            raise ValueError("invalid_template_length")
        nonce = os.urandom(12)
        sealed = nonce + self.cipher.encrypt(nonce, template, subject.encode())
        now = int(time.time())
        with self.connect() as db:
            db.execute(
                "INSERT INTO templates (subject, sealed, display_name, created_at, enrolled_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (subject, sealed, display_name, now, enrolled_by),
            )

    def delete(self, subject):
        with self.connect() as db:
            return bool(db.execute("DELETE FROM templates WHERE subject=?", (subject,)).rowcount)

    def audit(self, event_id, created, outcome):
        with self.connect() as db:
            db.execute("INSERT INTO audit (event_id, created, outcome) VALUES (?, ?, ?)",
                       (event_id, created, outcome))

    def pending(self, limit=25):
        with self.connect() as db:
            return db.execute("SELECT event_id, created, outcome FROM audit WHERE synced=0 ORDER BY created LIMIT ?",
                               (limit,)).fetchall()

    def mark_synced(self, event_id):
        with self.connect() as db:
            db.execute("UPDATE audit SET synced=1 WHERE event_id=?", (event_id,))
