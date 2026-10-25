"""
HarvestSight — Phase 1 / Script 02
Ring-search correction for SoilGrids urban-pixel nulls.

Reads datasets/fetch_summary.json (written by 01_fetch_datasets.py) to find
which states came back with null soil values — their capital-city coordinate
landed on a SoilGrids pixel masked as "urban" (capital cities being cities).

For each null state, searches outward in 4 rings (0.25°/0.5°/0.75°/1.0°,
approximated as ~28/55/83/111 km) at 4 cardinal offsets per ring, stopping
at the first point with real (non-null) soil data. Weather is then
re-fetched at that SAME corrected point, so weather and soil stay
co-located for that state rather than soil moving while weather stays
behind at the original, unusable point.

Run this AFTER 01_fetch_datasets.py, in the same environment.
Stdlib only — no installs needed.
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
CORRECTIONS_PATH = os.path.join(DATASETS_DIR, "coordinate_corrections.json")

# Must match 01_fetch_datasets.py exactly — copied rather than imported so
# this script can be handed off / run standalone without a package setup.
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

# Must match 01_fetch_datasets.py exactly.
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
INTER_REQUEST_DELAY_SEC = 1.5

# Ring search — degree offsets treated uniformly (~111km/degree), not a
# geodesic calculation. This is a simplification, not survey-grade.
RING_OFFSETS_DEG = [0.25, 0.5, 0.75, 1.0]
RING_APPROX_KM = {0.25: 28, 0.5: 55, 0.75: 83, 1.0: 111}
CARDINAL_DIRECTIONS = ["N", "S", "E", "W"]


def _fetch_json(url, label):
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HarvestSight/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_error = e
            print(f"    [RETRY {attempt}/{RETRY_ATTEMPTS}] {label} failed: {e}")
            time.sleep(RETRY_DELAY_SEC)
    print(f"    [FAIL] {label} — giving up. Last error: {last_error}")
    return None


def fetch_soilgrids(lat, lon):
    query = [("lon", lon), ("lat", lat)]
    for prop in SOIL_PROPERTIES:
        query.append(("property", prop))
    query.append(("depth", SOIL_DEPTH))
    query.append(("value", "mean"))
    url = f"{SOILGRIDS_URL}?{urllib.parse.urlencode(query)}"
    return _fetch_json(url, f"SoilGrids @ ({lat:.4f},{lon:.4f})")


def fetch_nasa_power(lat, lon):
    params = {
        "start": START_DATE, "end": END_DATE,
        "latitude": lat, "longitude": lon,
        "community": "AG", "parameters": NASA_POWER_PARAMS, "format": "JSON",
    }
    url = f"{NASA_POWER_URL}?{urllib.parse.urlencode(params)}"
    return _fetch_json(url, f"NASA POWER @ ({lat:.4f},{lon:.4f})")


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


def offset_point(lat, lon, degrees, direction):
    if direction == "N":
        return lat + degrees, lon
    if direction == "S":
        return lat - degrees, lon
    if direction == "E":
        return lat, lon + degrees
    if direction == "W":
        return lat, lon - degrees
    raise ValueError(direction)


def ring_search(orig_lat, orig_lon):
    """Try each ring/direction combo until real soil data is found."""
    for degrees in RING_OFFSETS_DEG:
        for direction in CARDINAL_DIRECTIONS:
            cand_lat, cand_lon = offset_point(orig_lat, orig_lon, degrees, direction)
            print(f"    trying {direction} ring {degrees}\u00b0 (~{RING_APPROX_KM[degrees]}km) "
                  f"-> ({cand_lat:.4f}, {cand_lon:.4f})")
            soil_json = fetch_soilgrids(cand_lat, cand_lon)
            time.sleep(INTER_REQUEST_DELAY_SEC)
            if not soil_response_is_null(soil_json):
                print(f"    [FOUND] real soil data at {direction} ring {degrees}\u00b0")
                return cand_lat, cand_lon, degrees, direction, soil_json
    return None  # exhausted all 16 candidates


def main():
    if not os.path.exists(SUMMARY_PATH):
        print(f"[ERROR] {SUMMARY_PATH} not found — run 01_fetch_datasets.py first.")
        return

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    null_states = summary.get("soil_null", [])
    if not null_states:
        print("No states flagged with null soil data — nothing to do.")
        return

    print(f"Ring-searching {len(null_states)} state(s): {null_states}")
    print(f"Worst case: {len(null_states) * len(RING_OFFSETS_DEG) * len(CARDINAL_DIRECTIONS)} "
          f"SoilGrids calls if none resolve early — budget a few minutes.\n")

    corrections = {}
    unresolved = []

    for state in null_states:
        orig_lat, orig_lon = STATE_COORDS[state]
        print(f"--- {state} (original: {orig_lat}, {orig_lon}) ---")
        result = ring_search(orig_lat, orig_lon)

        if result is None:
            n_candidates = len(RING_OFFSETS_DEG) * len(CARDINAL_DIRECTIONS)
            print(f"  [UNRESOLVED] {state} — no real soil data found across all "
                  f"{n_candidates} candidates. Flagging for manual review.")
            unresolved.append(state)
            continue

        cand_lat, cand_lon, degrees, direction, soil_json = result
        safe_name = state.replace(" ", "_")

        soil_path = os.path.join(DATASETS_DIR, f"soil_{safe_name}.json")
        with open(soil_path, "w", encoding="utf-8") as f:
            json.dump(soil_json, f)

        print(f"  re-fetching weather at corrected point for {state}...")
        weather_json = fetch_nasa_power(cand_lat, cand_lon)
        weather_refetched = weather_json is not None
        if weather_refetched:
            weather_path = os.path.join(DATASETS_DIR, f"weather_{safe_name}.json")
            with open(weather_path, "w", encoding="utf-8") as f:
                json.dump(weather_json, f)
            print(f"  [OK] weather re-fetched and saved for {state}")
        else:
            print(f"  [FAIL] weather re-fetch failed for {state} — soil is corrected "
                  f"but weather is still at the original point. MUST fix before Phase 5.")

        corrections[state] = {
            "original": {"lat": orig_lat, "lon": orig_lon},
            "corrected": {"lat": cand_lat, "lon": cand_lon},
            "ring_degrees": degrees,
            "ring_approx_km": RING_APPROX_KM[degrees],
            "direction": direction,
            "weather_refetched": weather_refetched,
        }
        time.sleep(INTER_REQUEST_DELAY_SEC)

    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump({"corrected": corrections, "unresolved": unresolved}, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Corrected: {len(corrections)} -> {list(corrections.keys())}")
    print(f"Unresolved: {len(unresolved)} -> {unresolved}")
    print(f"Corrections log written to {CORRECTIONS_PATH}")
    if unresolved:
        print("\nFor unresolved states: widen the ring further by hand, or pick a")
        print("different nearby city manually — don't silently proceed with null soil.")


if __name__ == "__main__":
    main()