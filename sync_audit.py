"""Upload minimal queued events to the IAM-protected AWS route with bounded retry."""
import argparse
import json
import logging
import time
from urllib.parse import urlsplit

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from storage import Store

LOGGER = logging.getLogger("liveguard.sync")
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3


def send_with_retry(client, url, data, creds, region):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = AWSRequest(method="POST", url=url, data=data,
                                 headers={"Content-Type": "application/json"})
            SigV4Auth(creds.get_frozen_credentials(), "execute-api", region).add_auth(request)
            response = client.post(url, content=data, headers=dict(request.headers))
            if response.status_code == 204:
                return True
            if response.status_code in TRANSIENT_STATUS_CODES:
                LOGGER.warning("Transient HTTP %d on attempt %d/%d", response.status_code, attempt, MAX_RETRIES)
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            raise RuntimeError(f"Unexpected audit response status: {response.status_code}")
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            LOGGER.warning("Transport error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            last_exc = exc
            time.sleep(0.5 * (2 ** (attempt - 1)))

    if last_exc:
        raise last_exc
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    url = urlsplit(args.url)
    suffix = f".execute-api.{args.region}.amazonaws.com"
    if (url.scheme != "https" or not url.hostname or not url.hostname.endswith(suffix)
            or url.path != "/audit" or url.netloc != url.hostname or url.query or url.fragment):
        parser.error("Expected the regional API Gateway HTTPS AuditUrl output")
    creds = boto3.Session(region_name=args.region).get_credentials()
    if creds is None:
        parser.error("No AWS credentials available")
    store = Store()
    count = 0
    with httpx.Client(timeout=5) as client:
        while rows := store.pending():
            for event_id, created, outcome in rows:
                data = json.dumps({"event_id": event_id, "created": created, "outcome": outcome})
                if send_with_retry(client, args.url, data, creds, args.region):
                    store.mark_synced(event_id)
                    count += 1
                else:
                    LOGGER.error("Failed to sync event %s after retries; aborting batch to preserve queue", event_id)
                    print(f"Synced {count} audit event(s) before retry limit reached")
                    return
    print(f"Synced {count} audit event(s)")


if __name__ == "__main__":
    main()
