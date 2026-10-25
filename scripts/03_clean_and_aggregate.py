"""
HarvestSight — Phase 2 / Script 03
Clean and aggregate the district-level yield CSV into one production-
weighted yield per (state, crop, season, year) — the unit everything
downstream (crop shortlist, risk labels, weekly sequences, training)
actually operates on.

Input:  datasets/crop-wise-area-production-yield.csv  (district-level,
        ~455k rows, one row per district x crop x season x year)
Output: cleaned_data/state_crop_season_year_yield.json (one entry per
        unique state/crop/season/year combination)

Filters, in order (a row is dropped by the FIRST check it fails, so the
four drop buckets below are mutually exclusive and should sum exactly
to total_rows - kept_rows):
  1. Season    -> keep only Kharif / Rabi / Summer (whitespace-stripped
                  before comparing; this dataset is known to sometimes
                  pad season values with trailing spaces). Everything
                  else ("Total" rollup, Autumn, Winter, anything else)
                  drops here.
  2. State     -> keep only the 28 target states (confirmed verbatim
                  against this CSV in Phase 1). UT rows drop here.
  3. Core fields -> missing state/crop/season/year, non-numeric or
                  negative area/production/yield, or area <= 0 (can't
                  compute a per-row contribution with zero area).
  4. Year      -> crop-year start outside 1997-2019 (crop-year
                  "2019-2020" is the last one kept, capped to match the
                  already-fetched weather range - confirmed choice, not
                  a data limitation).

Zero-yield rows that pass all four checks ARE kept - a real recorded 0
is crop-failure signal, not missing data. The Telangana/Andhra Pradesh
pre-2014 split is a known, unpatched limitation - no code here works
around it.

Aggregation: for each surviving (state, crop, season, year) group,
yield = sum(production) / sum(area) across all contributing district
rows - production-weighted, not a plain average of per-district yields.

Stdlib only (csv, json, os). No installs needed.
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")

YIELD_CSV = os.path.join(DATASETS_DIR, "crop-wise-area-production-yield.csv")
OUTPUT_PATH = os.path.join(CLEANED_DIR, "state_crop_season_year_yield.json")

# ---------------------------------------------------------------------------
# 28 target states - keys only (no coordinates needed here). Must match the
# state list used in 01_fetch_datasets.py / 02_retry_soil_weather.py exactly.
# ---------------------------------------------------------------------------
TARGET_STATES = frozenset([
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
])
assert len(TARGET_STATES) == 28, f"Expected 28 states, got {len(TARGET_STATES)}"

KEEP_SEASONS = frozenset(["Kharif", "Rabi", "Summer"])

# Crop-year cap - confirmed choice: cap at 2019-2020 to match the already-
# fetched weather range (through 2020-06-30), rather than extend the whole
# pipeline to whatever range the fresh CSV actually covers.
YEAR_MIN = 1997
YEAR_MAX = 2019

REQUIRED_COLUMNS = ["state_name", "crop_name", "season", "year", "area", "production", "yield"]
OPTIONAL_UNIT_COLUMNS = ["area_unit", "production_unit", "yield_unit"]

PROGRESS_EVERY = 100_000


def verify_columns(fieldnames):
    """
    Fail loudly, before processing a single row, if a required column is
    missing or misspelled - the same failure class as the bug that
    silently dropped 100% of rows in the original build (Title-Case
    assumed, lowercase real).
    """
    if fieldnames is None:
        print("[ERROR] Could not read a header row from the CSV at all.")
        return False

    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        print(f"[ERROR] Required column(s) not found in CSV header: {missing}")
        print(f"        Actual columns found: {fieldnames}")
        print("        Fix the column name mapping above before re-running -")
        print("        proceeding with a mismatch would silently drop rows.")
        return False

    print(f"[VERIFY][OK] All {len(REQUIRED_COLUMNS)} required columns found.")
    missing_optional = [c for c in OPTIONAL_UNIT_COLUMNS if c not in fieldnames]
    if missing_optional:
        print(f"[VERIFY] Optional unit columns not found (uniformity check will "
              f"be skipped for these): {missing_optional}")
    return True


def try_parse_float(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def try_parse_year_start(value):
    """Crop-year strings look like '1997-1998' - the start year is the
    first 4 characters."""
    if value is None:
        return None
    value = value.strip()
    if len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)

    print("=" * 70)
    print("HarvestSight Phase 2 / Script 03 - clean & aggregate yield data")
    print("=" * 70)

    if not os.path.exists(YIELD_CSV):
        print(f"[ERROR] Yield CSV not found at {YIELD_CSV}")
        return

    total_rows = 0
    kept_rows = 0
    dropped_season = 0
    dropped_state_not_in_28 = 0
    dropped_missing_core_fields = 0
    dropped_year_out_of_range = 0

    season_counts = {}          # every raw season value seen (kept + dropped)
    non28_state_counts = {}     # state values that failed the state filter
    unit_values = {"area_unit": set(), "production_unit": set(), "yield_unit": set()}

    # groups: (state, crop, season, year_start) -> {"total_area", "total_production", "n_districts"}
    groups = {}

    with open(YIELD_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not verify_columns(reader.fieldnames):
            return

        has_unit_cols = all(c in reader.fieldnames for c in OPTIONAL_UNIT_COLUMNS)

        for row in reader:
            total_rows += 1
            if total_rows % PROGRESS_EVERY == 0:
                print(f"  ...{total_rows:,} rows processed")

            # --- 1. Season ---
            season_raw = (row.get("season") or "").strip()
            season_counts[season_raw] = season_counts.get(season_raw, 0) + 1
            if season_raw not in KEEP_SEASONS:
                dropped_season += 1
                continue

            # --- 2. State ---
            state_raw = (row.get("state_name") or "").strip()
            if state_raw not in TARGET_STATES:
                dropped_state_not_in_28 += 1
                non28_state_counts[state_raw] = non28_state_counts.get(state_raw, 0) + 1
                continue

            # --- 3. Core fields ---
            crop_raw = (row.get("crop_name") or "").strip()
            year_raw = row.get("year")

            area = try_parse_float(row.get("area"))
            production = try_parse_float(row.get("production"))
            yield_val = try_parse_float(row.get("yield"))

            if (not crop_raw
                    or not year_raw
                    or area is None or area <= 0
                    or production is None or production < 0
                    or yield_val is None or yield_val < 0):
                dropped_missing_core_fields += 1
                continue

            # --- 4. Year range ---
            year_start = try_parse_year_start(year_raw)
            if year_start is None:
                dropped_missing_core_fields += 1
                continue
            if year_start < YEAR_MIN or year_start > YEAR_MAX:
                dropped_year_out_of_range += 1
                continue

            # --- Passed everything - accumulate into its group ---
            key = (state_raw, crop_raw, season_raw, year_start)
            if key not in groups:
                groups[key] = {"total_area": 0.0, "total_production": 0.0, "n_districts": 0}
            groups[key]["total_area"] += area
            groups[key]["total_production"] += production
            groups[key]["n_districts"] += 1
            kept_rows += 1

            if has_unit_cols:
                for col in OPTIONAL_UNIT_COLUMNS:
                    v = (row.get(col) or "").strip()
                    if v:
                        unit_values[col].add(v)

    # --- Sanity check: buckets should be mutually exclusive and complete ---
    accounted_for = (kept_rows + dropped_season + dropped_state_not_in_28
                      + dropped_missing_core_fields + dropped_year_out_of_range)
    if accounted_for != total_rows:
        print(f"[WARN] Row accounting mismatch: {accounted_for:,} accounted for vs "
              f"{total_rows:,} total read. This shouldn't happen with mutually "
              f"exclusive checks - investigate before trusting the output.")

    # --- Build output records ---
    records = []
    skipped_zero_area_groups = 0
    for (state, crop, season, year), agg in groups.items():
        if agg["total_area"] <= 0:
            # Shouldn't happen (area <= 0 rows are filtered per-row above),
            # but guard against it rather than dividing by zero silently.
            skipped_zero_area_groups += 1
            continue
        records.append({
            "state": state,
            "crop": crop,
            "season": season,
            "year": year,
            "yield": agg["total_production"] / agg["total_area"],
            "total_area": agg["total_area"],
            "total_production": agg["total_production"],
            "n_districts": agg["n_districts"],
        })

    records.sort(key=lambda r: (r["state"], r["crop"], r["season"], r["year"]))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    distinct_crops = len(set(r["crop"] for r in records))

    # --- Report ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"total_rows:                  {total_rows:,}")
    print(f"kept_rows:                   {kept_rows:,}")
    print(f"dropped_season:              {dropped_season:,}")
    print(f"dropped_state_not_in_28:     {dropped_state_not_in_28:,}")
    print(f"dropped_missing_core_fields: {dropped_missing_core_fields:,}")
    print(f"dropped_year_out_of_range:   {dropped_year_out_of_range:,}")
    if skipped_zero_area_groups:
        print(f"[WARN] skipped_zero_area_groups: {skipped_zero_area_groups} "
              f"(unexpected - investigate)")
    print(f"\n-> {len(records):,} state/crop/season/year cells, "
          f"{distinct_crops} distinct crops")
    print(f"   (benchmark from the prior validated run on this same source: "
          f"252,938 kept rows -> 15,185 cells, 77 crops - expect a fresh "
          f"download to land close to this, not wildly off)")

    print("\nSeason value distribution (all rows, before filtering):")
    for season, count in sorted(season_counts.items(), key=lambda kv: -kv[1]):
        tag = "KEPT" if season in KEEP_SEASONS else "dropped"
        print(f"  {season!r:20s} {count:>8,}  [{tag}]")

    if non28_state_counts:
        print("\nState values excluded as non-target (expected: the ~6 UT entries):")
        for state, count in sorted(non28_state_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {state!r:45s} {count:>8,}")

    if has_unit_cols:
        print("\nUnit uniformity check (among kept rows):")
        for col, values in unit_values.items():
            if len(values) <= 1:
                print(f"  {col}: {sorted(values)} [OK - uniform]")
            else:
                print(f"  {col}: {sorted(values)} [WARNING - NOT uniform, "
                      f"aggregation assumes matching units across rows]")

    print(f"\nOutput written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()