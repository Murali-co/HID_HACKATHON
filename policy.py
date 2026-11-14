"""Locally verified policy snapshot for the demo kiosk."""
import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

STATE = Path(os.environ.get("LIVEGUARD_STATE", "state"))
DOOR = "campus-lab"
MAX_TTL = 86_400


def encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def install(destination, payload):
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_suffix(".tmp")
    with open(temporary, "wb") as file:
        file.write(payload)
    os.replace(temporary, destination)


def evaluate(subject, now, state=STATE):
    try:
        public = serialization.load_pem_public_key((state / "policy.pub").read_bytes())
        signed = json.loads((state / "policy.json").read_bytes())
        policy = signed["policy"]
        public.verify(base64.b64decode(signed["signature"], validate=True), encoded(policy))
        issued, expires = policy["issued"], policy["expires"]
        if policy["door"] != DOOR or not issued <= now < expires or not 0 < expires - issued <= MAX_TTL:
            return "policy_unavailable"
        window = policy["access"].get(subject)
        if window is None:
            return "outside_schedule"
        local = datetime.fromtimestamp(now, ZoneInfo(policy["timezone"]))
        start, end = window
        if not 0 <= start < end <= 1440:
            return "policy_unavailable"
        return "authorized" if start <= local.hour * 60 + local.minute < end else "outside_schedule"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, InvalidSignature, ZoneInfoNotFoundError):
        return "policy_unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signer-dir", required=True, type=Path)
    parser.add_argument("--subject", action="append", default=[], help="Repeat for each allowed ID")
    parser.add_argument("--merge", action="store_true", help="Merge with existing valid policy subjects")
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--start-minute", type=int, default=0)
    parser.add_argument("--end-minute", type=int, default=1440)
    parser.add_argument("--ttl", type=int, default=28800)
    args = parser.parse_args()
    if not 0 < args.ttl <= MAX_TTL or not 0 <= args.start_minute < args.end_minute <= 1440:
        parser.error("Invalid policy lifetime or same-day access window")
    ZoneInfo(args.timezone)
    args.signer_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_path = args.signer_dir / "policy.key"
    try:
        key = Ed25519PrivateKey.generate()
        with os.fdopen(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as file:
            file.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    except FileExistsError:
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    now = int(time.time())

    access_map = {}
    if args.merge and (STATE / "policy.json").is_file():
        try:
            old_signed = json.loads((STATE / "policy.json").read_bytes())
            old_policy = old_signed.get("policy", {})
            if old_policy.get("door") == DOOR and isinstance(old_policy.get("access"), dict):
                for subj, win in old_policy["access"].items():
                    if isinstance(win, list) and len(win) == 2:
                        access_map[subj] = win
        except Exception:
            pass

    for s in args.subject:
        access_map[s] = [args.start_minute, args.end_minute]

    policy = {"issued": now, "expires": now + args.ttl, "door": DOOR, "timezone": args.timezone,
              "access": access_map}
    envelope = {"policy": policy, "signature": base64.b64encode(key.sign(encoded(policy))).decode()}
    install(STATE / "policy.pub", key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    install(STATE / "policy.json", encoded(envelope))
    subject_list = ", ".join(sorted(access_map.keys())) or "none"
    print(f"Issued {args.ttl}s demo policy for {DOOR} with {len(access_map)} subject(s): {subject_list}")


if __name__ == "__main__":
    main()
