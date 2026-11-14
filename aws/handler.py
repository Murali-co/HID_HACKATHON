import base64
import json
import os
import time
import uuid

import boto3
from botocore.exceptions import ClientError


def ingest(event, context):
    try:
        raw_body = event.get("body") or ""
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        body = json.loads(raw_body)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON object body")
        if set(body) != {"event_id", "created", "outcome"}:
            raise ValueError("Unexpected fields")
        if str(uuid.UUID(body["event_id"], version=4)) != body["event_id"]:
            raise ValueError("Invalid event ID")
        if type(body["created"]) is not int or not 0 < body["created"] <= time.time() + 300:
            raise ValueError("Invalid timestamp")
        if body["outcome"] not in ("ALLOW", "DENY", "REQUEST_VERIFICATION"):
            raise ValueError("Invalid outcome")
    except (ValueError, TypeError, AttributeError):
        return {"statusCode": 400, "body": '{"error":"invalid_event"}'}
    table = boto3.resource("dynamodb").Table(os.environ["TABLE"])
    try:
        table.put_item(Item={**body, "expires": int(time.time()) + 7 * 86400},
                       ConditionExpression="attribute_not_exists(event_id)")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
    return {"statusCode": 204, "body": ""}
