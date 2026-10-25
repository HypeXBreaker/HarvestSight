"""
HarvestSight — Phase 1 / Script 01
Fetch weather (NASA POWER Daily Point API) and soil (SoilGrids v2.0) data
for 28 representative Indian state points.

The crop yield CSV (crop-wise-area-production-yield.csv) is downloaded
MANUALLY via Chrome from India Data Portal and is expected to already be
sitting in datasets/ — this script does not fetch it, only reads it once
for a sanity check (see verify_state_names_against_csv).

Requirements: Python 3.x, stdlib only (urllib, json, os, time, csv).
No installs needed. Must run somewhere with open internet access
(local machine or Google Colab) — NASA POWER and SoilGrids are both
free/keyless public APIs.

Expected folder layout (paths below are resolved relative to this
script's own location, not the current working directory, so it
doesn't matter where you invoke `python` from):

    HarvestSight/
    ├── datasets/        <- this script reads/writes here
    ├── cleaned_data/
    └── scripts/         <- this script lives here

If you're running this in Colab instead of locally, upload/mount the
whole HarvestSight/ folder (not just this file) so the relative path
below still resolves — or hardcode DATASETS_DIR to an absolute path.
"""

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script's own location, not cwd.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # HarvestSight/
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
YIELD_CSV = os.path.join(DATASETS_DIR, "crop-wise-area-production-yield.csv")

# ---------------------------------------------------------------------------
# 28 target states -> representative (lat, lon).
# Capital city for each state, EXCEPT Punjab and Haryana, which both share
# Chandigarh as capital — using Chandigarh for both would give them near-
# identical weather and defeat the purpose of a per-state model, so each
# uses its own agricultural hub city instead.
# ---------------------------------------------------------------------------
STATE_COORDS = {
    "Andhra Pradesh":    (16.5062, 80.6480),   # Amaravati
    "Arunachal Pradesh": (27.0844, 93.6053),   # Itanagar
    "Assam":             (26.1433, 91.7898),   # Dispur/Guwahati
    "Bihar":             (25.5941, 85.1376),   # Patna
    "Chhattisgarh":      (21.2514, 81.6296),   # Raipur
    "Goa":               (15.4909, 73.8278),   # Panaji
    "Gujarat":           (23.2156, 72.6369),   # Gandhinagar
    "Haryana":           (29.1492, 75.7217),   # Hisar (exception)
    "Himachal Pradesh":  (31.1048, 77.1734),   # Shimla
    "Jharkhand":         (23.3441, 85.3096),   # Ranchi
    "Karnataka":         (12.9716, 77.5946),   # Bengaluru
    "Kerala":            (8.5241, 76.9366),    # Thiruvananthapuram
    "Madhya Pradesh":    (23.2599, 77.4126),   # Bhopal
    "Maharashtra":       (19.0760, 72.8777),   # Mumbai
    "Manipur":           (24.8170, 93.9368),   # Imphal
    "Meghalaya":         (25.5788, 91.8933),   # Shillong
    "Mizoram":           (23.7271, 92.7176),   # Aizawl
    "Nagaland":          (25.6751, 94.1086),   # Kohima
    "Odisha":            (20.2961, 85.8245),   # Bhubaneswar
    "Punjab":            (30.9010, 75.8573),   # Ludhiana (exception)
    "Rajasthan":         (26.9124, 75.7873),   # Jaipur
    "Sikkim":            (27.3389, 88.6065),   # Gangtok
    "Tamil Nadu":        (13.0827, 80.2707),   # Chennai
    "Telangana":         (17.3850, 78.4867),   # Hyderabad
    "Tripura":           (23.8315, 91.2868),   # Agartala
    "Uttar Pradesh":     (26.8467, 80.9462),   # Lucknow
    "Uttarakhand":       (30.3165, 78.0322),   # Dehradun
    "West Bengal":       (22.5726, 88.3639),   # Kolkata
}
assert len(STATE_COORDS) == 28, f"Expected 28 states, got {len(STATE_COORDS)}"

# ---------------------------------------------------------------------------
# Weather date range.
# Yield data runs crop-years 1997-1998 .. 2019-2020 (confirmed in the prior
# build). Rabi crosses into the following calendar year (Oct 1 -> Mar 31)
# and Summer for a "2019-2020" crop-year record falls Mar-Jun 2020, so the
# end date is padded to cover that fully.
#
# ASSUMPTION FLAGGED: this end date is carried over from the prior build's
# real yield data. Since this is a freshly re-downloaded CSV, double-check
# its actual max crop-year (e.g. `df["year"].max()` in pandas, or
# `sort -u` on the year column) before trusting this cutoff — if the fresh
# download covers more recent years, this needs to move out to match.
# ---------------------------------------------------------------------------
START_DATE = "19970101"
END_DATE = "20200630"

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,GWETROOT"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["phh2o", "soc", "clay"]
SOIL_DEPTH = "0-5cm"

REQUEST_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_DELAY_SEC = 5
INTER_REQUEST_DELAY_SEC = 1.5  # be polite to both free APIs


def verify_state_names_against_csv():
    """
    Cross-check STATE_COORDS keys against the real, distinct state_name
    values in the yield CSV — verbatim, case-sensitive.

    This is the same failure class as Phase 2's column-name bug (title-case
    assumed, lowercase real, 100% of rows silently dropped). Catching a
    state-name spelling mismatch HERE, before any API calls, is free;
    catching it after 28 states' worth of weather/soil data is already
    downloaded means redoing the fetch.
    """
    if not os.path.exists(YIELD_CSV):
        print(f"[WARN] Yield CSV not found at {YIELD_CSV}")
        print("       Skipping verification — proceeding on the assumption")
        print("       that STATE_COORDS keys are correct. Fix this before")
        print("       trusting Phase 2's join.")
        return

    csv_states = set()
    with open(YIELD_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "state_name" not in reader.fieldnames:
            print(f"[WARN] 'state_name' column not found. Actual columns: {reader.fieldnames}")
            print("       Skipping verification.")
            return
        for row in reader:
            csv_states.add(row["state_name"])

    expected = set(STATE_COORDS.keys())
    missing_from_csv = expected - csv_states
    extra_in_csv = csv_states - expected

    print(f"[VERIFY] {len(csv_states)} distinct state_name values found in CSV.")
    if missing_from_csv:
        print(f"[VERIFY][MISMATCH] {len(missing_from_csv)} STATE_COORDS "
              f"entries NOT found verbatim in the CSV: {sorted(missing_from_csv)}")
        print("          Fix the spelling in STATE_COORDS before trusting any")
        print("          downstream join — a mismatch here means that state's")
        print("          weather/soil silently won't join to its yield rows later.")
    else:
        print("[VERIFY][OK] All 28 STATE_COORDS keys matched verbatim in the CSV.")

    print(f"[VERIFY] {len(extra_in_csv)} distinct CSV state_name values are NOT "
          f"in STATE_COORDS (expected — these should be the ~6 excludable UT "
          f"entries): {sorted(extra_in_csv)}")


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


def fetch_nasa_power(label, lat, lon):
    params = {
        "start": START_DATE,
        "end": END_DATE,
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": NASA_POWER_PARAMS,
        "format": "JSON",
    }
    url = f"{NASA_POWER_URL}?{urllib.parse.urlencode(params)}"
    return _fetch_json(url, f"NASA POWER weather for {label}")


def fetch_soilgrids(label, lat, lon):
    query = [("lon", lon), ("lat", lat)]
    for prop in SOIL_PROPERTIES:
        query.append(("property", prop))
    query.append(("depth", SOIL_DEPTH))
    query.append(("value", "mean"))
    url = f"{SOILGRIDS_URL}?{urllib.parse.urlencode(query)}"
    return _fetch_json(url, f"SoilGrids soil for {label}")


def soil_response_is_null(soil_json):
    """
    Detect the urban-pixel-mask case: every requested property's mean value
    at the requested depth comes back None/missing.

    ASSUMPTION FLAGGED: this reads properties.layers[].depths[].values.mean,
    per SoilGrids v2.0's documented response shape. Like Phase 5's weather/
    soil parsing, this has NOT been checked against a real response yet —
    the first real console output from this script is what confirms or
    breaks that assumption.
    """
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
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print("=" * 70)
    print("HarvestSight Phase 1 / Script 01 — fetch weather + soil per state")
    print("=" * 70)

    verify_state_names_against_csv()
    print()

    summary = {"weather_ok": [], "weather_fail": [], "soil_ok": [], "soil_null": [], "soil_fail": []}

    for state, (lat, lon) in STATE_COORDS.items():
        print(f"\n--- {state} ({lat}, {lon}) ---")
        safe_name = state.replace(" ", "_")

        # Weather
        weather_data = fetch_nasa_power(state, lat, lon)
        if weather_data is not None:
            weather_path = os.path.join(DATASETS_DIR, f"weather_{safe_name}.json")
            with open(weather_path, "w", encoding="utf-8") as f:
                json.dump(weather_data, f)
            print(f"  [OK] weather -> {weather_path}")
            summary["weather_ok"].append(state)
        else:
            summary["weather_fail"].append(state)
        time.sleep(INTER_REQUEST_DELAY_SEC)

        # Soil
        soil_data = fetch_soilgrids(state, lat, lon)
        if soil_data is not None:
            soil_path = os.path.join(DATASETS_DIR, f"soil_{safe_name}.json")
            with open(soil_path, "w", encoding="utf-8") as f:
                json.dump(soil_data, f)
            if soil_response_is_null(soil_data):
                print(f"  [NULL] soil values all null for {state} (likely urban-pixel "
                      f"mask) — flagged for script 02's ring-search retry.")
                summary["soil_null"].append(state)
            else:
                print(f"  [OK] soil -> {soil_path}")
                summary["soil_ok"].append(state)
        else:
            summary["soil_fail"].append(state)
        time.sleep(INTER_REQUEST_DELAY_SEC)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, states in summary.items():
        print(f"{key}: {len(states)} -> {states}")

    # Written so script 02 can pick up the soil_null list directly, instead
    # of re-deriving it by re-reading every soil file.
    summary_path = os.path.join(DATASETS_DIR, "fetch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print("Next: run 02_retry_soil_weather.py to fix any soil_null states.")


if __name__ == "__main__":
    main()