"""Summarize labeled kiosk CSV attempts without dropping uncertain results."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict

ATTACKS = {"print", "screen_photo", "video_replay", "mask"}
VALID_OUTCOMES = {"ALLOW", "DENY", "REQUEST_VERIFICATION"}
VALID_PAD = {"LIVE", "SPOOF", "UNCERTAIN"}
VALID_ATTEMPTS = {"initial", "retake"}


def ratio(count, total):
    return round(count / total, 4) if total else None


def percentile(values, fraction):
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = math.ceil(fraction * len(sorted_vals)) - 1
    return round(sorted_vals[idx], 1)


def parse_float(val):
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("none", "nan", "null"):
        return None
    try:
        f = float(val_str)
        return f if math.isfinite(f) and f >= 0 else None
    except (ValueError, TypeError):
        return None


def summarize(rows):
    scenarios = defaultdict(list)
    rejected_rows = 0

    for row in rows:
        scen = (row.get("scenario") or "").strip()
        if not scen:
            continue
        scenarios[scen].append(row)

    report = {}
    for scenario, attempts in scenarios.items():
        initial = [row for row in attempts if (row.get("attempt") or "").strip() == "initial"]
        retakes = [row for row in attempts if (row.get("attempt") or "").strip() == "retake"]

        counts = Counter((row.get("outcome") or "").strip() for row in initial)
        pad = Counter((row.get("pad") or "").strip() for row in initial)

        # Collect validated finite non-negative timings
        browser_times = []
        for row in initial:
            val = parse_float(row.get("browser_ms"))
            if val is not None:
                browser_times.append(val)

        server_times = []
        for row in initial:
            val = parse_float(row.get("server_ms"))
            if val is not None:
                server_times.append(val)

        item = {
            "initial_attempts": len(initial),
            "retakes": len(retakes),
            "retake_outcomes": dict(Counter((row.get("outcome") or "").strip() for row in retakes)),
            "outcomes": dict(counts),
            "pad_outcomes": dict(pad),
            "browser_p50_ms": percentile(browser_times, 0.50),
            "browser_p95_ms": percentile(browser_times, 0.95),
            "browser_max_ms": round(max(browser_times), 1) if browser_times else None,
            "server_p95_ms": percentile(server_times, 0.95),
            "measured_browser_timings": len(browser_times),
            "within_2s": sum(value <= 2000.0 for value in browser_times),
            "over_2s": sum(value > 2000.0 for value in browser_times),
            "request_verification_rate": ratio(counts.get("REQUEST_VERIFICATION", 0), len(initial)),
            "pad_uncertain_rate": ratio(pad.get("UNCERTAIN", 0), len(initial)),
        }

        if scenario in ATTACKS:
            item["attack_pad_pass_rate"] = ratio(pad.get("LIVE", 0), len(initial))
            item["attack_pad_detection_rate"] = ratio(pad.get("SPOOF", 0), len(initial))
            item["attack_access_allow_rate"] = ratio(counts.get("ALLOW", 0), len(initial))

        if scenario == "genuine":
            item["genuine_pad_spoof_rate"] = ratio(pad.get("SPOOF", 0), len(initial))
            item["genuine_access_deny_rate"] = ratio(counts.get("DENY", 0), len(initial))
            item["genuine_access_non_allow_rate"] = ratio(len(initial) - counts.get("ALLOW", 0), len(initial))

        if scenario == "wrong_identity":
            item["wrong_identity_access_allow_rate"] = ratio(counts.get("ALLOW", 0), len(initial))

        report[scenario] = item

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as file:
        print(json.dumps(summarize(csv.DictReader(file)), indent=2))
