"""
HarvestSight — Phase 1 / Script 01b
Targeted retry for states in fetch_summary.json's soil_fail list.

soil_fail means the SoilGrids request never completed (timeout / network
error) — unlike soil_null, which means the request succeeded and returned
real "no data at this pixel" nulls. A fail tells you nothing about whether
that state's point is an urban pixel or not, so it needs to be resolved
one way or the other before Phase 5, not left as a gap.

This script ONLY retries soil for states in soil_fail. It does not touch
weather (already fetched successfully) or any other state.

After running, fetch_summary.json is updated in place:
  - real soil data came back -> moved into soil_ok
  - request succeeded but values are null -> moved into soil_null
    (so 02_retry_soil_weather.py's ring-search will pick it up)
  - still fails -> stays in soil_fail, printed for manual attention

Run this BEFORE 02_retry_soil_weather.py.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
SUMMARY_PATH = os.path.join(DATASETS_DIR, "fetch_summary.json")

# Must match 01_fetch_datasets.py exactly.
STATE_COORDS = {
    "Andhra Pradesh":    (16.5062, 80.6480),
    "Arunachal Pradesh": (27.0844, 93.6053),
    "Assam":             (26.1433, 91.7898),
    "Bihar":             (25.5941, 85.1376),
    "Chhattisgarh":      (21.2514, 81.6296),
    "Goa":               (15.4909, 73.8278),
    "Gujarat":           (23.2156, 72.6369),
    "Haryana":           (29.1492, 75.7217),
    "Himachal Pradesh":  (31.1048, 77.1734),
    "Jharkhand":         (23.3441, 85.3096),
    "Karnataka":         (12.9716, 77.5946),
    "Kerala":            (8.5241, 76.9366),
    "Madhya Pradesh":    (23.2599, 77.4126),
    "Maharashtra":       (19.0760, 72.8777),
    "Manipur":           (24.8170, 93.9368),
    "Meghalaya":         (25.5788, 91.8933),
    "Mizoram":           (23.7271, 92.7176),
    "Nagaland":          (25.6751, 94.1086),
    "Odisha":            (20.2961, 85.8245),
    "Punjab":            (30.9010, 75.8573),
    "Rajasthan":         (26.9124, 75.7873),
    "Sikkim":            (27.3389, 88.6065),
    "Tamil Nadu":        (13.0827, 80.2707),
    "Telangana":         (17.3850, 78.4867),
    "Tripura":           (23.8315, 91.2868),
    "Uttar Pradesh":     (26.8467, 80.9462),
    "Uttarakhand":       (30.3165, 78.0322),
    "West Bengal":       (22.5726, 88.3639),
}

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["phh2o", "soc", "clay"]
SOIL_DEPTH = "0-5cm"

REQUEST_TIMEOUT = 90          # longer than 01's 60s — Tamil Nadu timed out 3x at 60s
RETRY_ATTEMPTS = 4
RETRY_DELAY_SEC = 8
INTER_REQUEST_DELAY_SEC = 1.5


def _fetch_json(url, label):
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HarvestSight/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_error = e
            print(f"  [RETRY {attempt}/{RETRY_ATTEMPTS}] {label} failed: {e}")
            time.sleep(RETRY_DELAY_SEC)
    print(f"  [FAIL] {label} — giving up after {RETRY_ATTEMPTS} attempts. Last error: {last_error}")
    return None


def fetch_soilgrids(lat, lon):
    query = [("lon", lon), ("lat", lat)]
    for prop in SOIL_PROPERTIES:
        query.append(("property", prop))
    query.append(("depth", SOIL_DEPTH))
    query.append(("value", "mean"))
    url = f"{SOILGRIDS_URL}?{urllib.parse.urlencode(query)}"
    return _fetch_json(url, f"SoilGrids soil for {lat},{lon}")


def soil_response_is_null(soil_json):
    if soil_json is None:
        return True
    try:
        layers = soil_json["properties"]["layers"]
    except (KeyError, TypeError):
        return True
    if not layers:
        return True
    for layer in layers:
        for depth_entry in layer.get("depths", []):
            if depth_entry.get("values", {}).get("mean") is not None:
                return False
    return True


def main():
    if not os.path.exists(SUMMARY_PATH):
        print(f"[ERROR] {SUMMARY_PATH} not found — run 01_fetch_datasets.py first.")
        return

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    fail_states = list(summary.get("soil_fail", []))
    if not fail_states:
        print("No states in soil_fail — nothing to retry.")
        return

    print(f"Retrying soil fetch for {len(fail_states)} state(s): {fail_states}\n")

    still_failing = []

    for state in fail_states:
        lat, lon = STATE_COORDS[state]
        print(f"--- {state} ({lat}, {lon}) ---")
        soil_data = fetch_soilgrids(lat, lon)

        if soil_data is None:
            print(f"  [STILL FAILING] {state} — leaving in soil_fail for manual attention.")
            still_failing.append(state)
            time.sleep(INTER_REQUEST_DELAY_SEC)
            continue

        safe_name = state.replace(" ", "_")
        soil_path = os.path.join(DATASETS_DIR, f"soil_{safe_name}.json")
        with open(soil_path, "w", encoding="utf-8") as f:
            json.dump(soil_data, f)

        if soil_response_is_null(soil_data):
            print(f"  [NULL] {state} came back with real response but null values "
                  f"(urban-pixel mask) — moved to soil_null for ring-search in 02.")
            summary["soil_null"].append(state)
        else:
            print(f"  [OK] {state} soil saved -> {soil_path}")
            summary["soil_ok"].append(state)

        time.sleep(INTER_REQUEST_DELAY_SEC)

    summary["soil_fail"] = still_failing

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Resolved: {len(fail_states) - len(still_failing)}")
    print(f"Still failing: {still_failing}")
    print(f"fetch_summary.json updated at {SUMMARY_PATH}")
    if still_failing:
        print("\nFor states still failing: check your network/VPN, or try again later —")
        print("this may be a transient SoilGrids server issue rather than anything in the code.")


if __name__ == "__main__":
    main()