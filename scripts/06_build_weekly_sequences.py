"""
06_build_weekly_sequences.py

HarvestSight -- Phase 5: Feature Engineering (Weather/Soil Join & Sequencing).

For every labeled (state, crop, season, year) record from Phase 4 -- excluding
`insufficient_data` rows, which have no usable target -- this slices that
season's calendar-date window out of the state's daily weather file, resamples
it into weekly buckets, and attaches the state's static soil chemistry. Output
is one fixed-length weekly sequence per record, ready for the RNN.

Schema note: parsing below was written against real sample files
(weather_Andhra_Pradesh.json, soil_Andhra_Pradesh.json), not just the
documented API format -- this closes the open risk flagged in thesis
Section 5.3 (structural assumptions never validated against a real file).
Confirmed real shapes:
  weather: properties.parameter.<PARAM>.<YYYYMMDD> -> float
           header.fill_value == -999.0 (sentinel for missing days)
  soil:    properties.layers[i].name / .unit_measure.d_factor /
           .depths[0].values.mean  (raw value; divide by d_factor for the
           true unit -- never hardcode pH/SOC/clay x10 factors)

Input:
  cleaned_data/yield_with_risk_labels.json   (Phase 4 output, 10,936 records)
  datasets/weather_<State_With_Underscores>.json   (28 states)
  datasets/soil_<State_With_Underscores>.json      (28 states)

Output:
  cleaned_data/rnn_sequences.json

Stdlib only.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"

LABELS_INPUT_PATH = CLEANED_DATA_DIR / "yield_with_risk_labels.json"
OUTPUT_PATH = CLEANED_DATA_DIR / "rnn_sequences.json"

# Order is fixed and used consistently for every weekly feature vector.
# Matches the six params actually fetched (Phase 1) and the real weather
# file's own `parameters` key order.
WEATHER_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
SUM_PARAMS = {"PRECTOTCORR"}  # summed weekly; everything else is averaged

FILL_SENTINEL_MAX = -900  # any daily value <= this is treated as missing (NASA POWER fill is -999)
EXPECTED_FILL_VALUE = -999.0  # sanity cross-check against each file's own header.fill_value

SOIL_LAYER_NAMES = {"phh2o", "soc", "clay"}
EXPECTED_SOIL_DEPTH_LABEL = "0-5cm"

# Season -> (start_month, start_day, end_month, end_day, crosses_into_next_year)
SEASON_WINDOWS = {
    "Kharif": (6, 1, 11, 30, False),
    "Rabi": (10, 1, 3, 31, True),
    "Summer": (3, 1, 6, 30, False),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def state_to_filename_stub(state):
    return state.replace(" ", "_")


def season_date_range(season, year):
    """Returns (start_date, end_date) for a (season, crop-year-start) pair."""
    start_month, start_day, end_month, end_day, crosses_year = SEASON_WINDOWS[season]
    start = date(year, start_month, start_day)
    end_year = year + 1 if crosses_year else year
    end = date(end_year, end_month, end_day)
    return start, end


def load_weather(state, cache, warnings):
    if state in cache:
        return cache[state]
    path = DATASETS_DIR / f"weather_{state_to_filename_stub(state)}.json"
    if not path.exists():
        cache[state] = None
        warnings.append(f"Missing weather file for '{state}': {path}")
        return None
    data = load_json(path)
    fill_value = data.get("header", {}).get("fill_value")
    if fill_value != EXPECTED_FILL_VALUE:
        warnings.append(
            f"'{state}' weather header.fill_value is {fill_value!r}, "
            f"expected {EXPECTED_FILL_VALUE} -- FILL_SENTINEL_MAX may need review"
        )
    cache[state] = data["properties"]["parameter"]
    return cache[state]


def load_soil(state, cache, warnings):
    if state in cache:
        return cache[state]
    path = DATASETS_DIR / f"soil_{state_to_filename_stub(state)}.json"
    if not path.exists():
        cache[state] = None
        warnings.append(f"Missing soil file for '{state}': {path}")
        return None
    data = load_json(path)
    soil = {}
    seen_names = set()
    for layer in data["properties"]["layers"]:
        name = layer["name"]
        seen_names.add(name)
        d_factor = layer["unit_measure"]["d_factor"]  # never hardcoded -- read per layer
        depth = layer["depths"][0]
        if depth["label"] != EXPECTED_SOIL_DEPTH_LABEL:
            warnings.append(
                f"'{state}' soil layer '{name}' depth label is "
                f"'{depth['label']}', expected '{EXPECTED_SOIL_DEPTH_LABEL}'"
            )
        raw_mean = depth["values"]["mean"]
        soil[name] = raw_mean / d_factor if d_factor else raw_mean
    if seen_names != SOIL_LAYER_NAMES:
        warnings.append(
            f"'{state}' soil layers {sorted(seen_names)} don't match "
            f"expected {sorted(SOIL_LAYER_NAMES)}"
        )
    cache[state] = soil
    return soil


def build_weekly_sequence(weather, start_date, end_date):
    """
    Splits [start_date, end_date] into consecutive 7-day buckets and
    aggregates each weather param per bucket (sum for rainfall, mean for
    everything else). A day whose value is <= FILL_SENTINEL_MAX, or a day
    missing from the file entirely, is excluded from that week's aggregate.
    A week where every day is missing for a param yields None for that
    param (flagged missing) rather than a silent 0.

    Returns (sequence, missing_week_param_count) where sequence is a list
    of weeks, each week a list of floats/None in WEATHER_PARAMS order.
    """
    total_days = (end_date - start_date).days + 1
    n_weeks = math.ceil(total_days / 7)

    sequence = []
    missing_week_param_count = 0

    for week_idx in range(n_weeks):
        week_start = start_date + timedelta(days=7 * week_idx)
        week_end = min(week_start + timedelta(days=6), end_date)

        week_vector = []
        for param in WEATHER_PARAMS:
            param_data = weather.get(param, {})
            valid_values = []
            day = week_start
            while day <= week_end:
                key = day.strftime("%Y%m%d")
                value = param_data.get(key)
                if value is not None and value > FILL_SENTINEL_MAX:
                    valid_values.append(value)
                day += timedelta(days=1)

            if not valid_values:
                week_vector.append(None)
                missing_week_param_count += 1
            elif param in SUM_PARAMS:
                week_vector.append(sum(valid_values))
            else:
                week_vector.append(sum(valid_values) / len(valid_values))

        sequence.append(week_vector)

    return sequence, missing_week_param_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading labeled records from: {LABELS_INPUT_PATH}")
    labeled_records = load_json(LABELS_INPUT_PATH)
    print(f"  {len(labeled_records):,} total labeled records (Phase 4 output)")

    scoped = [r for r in labeled_records if r["risk"] != "insufficient_data"]
    dropped = len(labeled_records) - len(scoped)
    print(f"Dropped insufficient_data: {dropped:,} -> {len(scoped):,} records to sequence")

    weather_cache = {}
    soil_cache = {}
    file_warnings = []

    # ---- Pass 1: build every record's raw (unpadded) weekly sequence ----
    built = []
    skipped_missing_files = 0
    total_missing_week_params = 0
    season_length_stats = {}  # season -> list of natural sequence lengths

    for r in scoped:
        state, crop, season, year = r["state"], r["crop"], r["season"], r["year"]

        weather = load_weather(state, weather_cache, file_warnings)
        soil = load_soil(state, soil_cache, file_warnings)
        if weather is None or soil is None:
            skipped_missing_files += 1
            continue

        start_date, end_date = season_date_range(season, year)
        sequence, missing_week_params = build_weekly_sequence(weather, start_date, end_date)
        total_missing_week_params += missing_week_params

        season_length_stats.setdefault(season, []).append(len(sequence))

        built.append({
            "state": state,
            "crop": crop,
            "season": season,
            "year": year,
            "yield": r["yield"],
            "risk": r["risk"],
            "risk_detail": r["risk_detail"],
            "soil": soil,
            "sequence": sequence,
            "sequence_length": len(sequence),
        })

    if skipped_missing_files:
        print(f"  WARNING: skipped {skipped_missing_files:,} records due to missing weather/soil files")
    for w in file_warnings:
        print(f"  WARNING: {w}")

    # ---- Determine the true global max sequence length from real data ----
    max_seq_len = max((rec["sequence_length"] for rec in built), default=0)
    print(f"\nMax sequence length found across all real (season, year) combinations: {max_seq_len} weeks")
    for season, lengths in sorted(season_length_stats.items()):
        print(f"  {season:<8} natural length range: {min(lengths)}-{max(lengths)} weeks "
              f"(n={len(lengths):,})")

    # ---- Pass 2: zero-pad every sequence up to max_seq_len ----
    zero_week = [0.0] * len(WEATHER_PARAMS)
    for rec in built:
        pad_count = max_seq_len - rec["sequence_length"]
        if pad_count > 0:
            rec["sequence"] = rec["sequence"] + [list(zero_week) for _ in range(pad_count)]

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(built, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 5 -- Weekly Sequence Build: Summary")
    print("=" * 60)
    print(f"Input labeled records:          {len(labeled_records):,}")
    print(f"Dropped (insufficient_data):     {dropped:,}")
    print(f"Skipped (missing files):         {skipped_missing_files:,}")
    print(f"Output sequence records:         {len(built):,}")
    print(f"Weather params per timestep:      {len(WEATHER_PARAMS)} ({', '.join(WEATHER_PARAMS)})")
    print(f"Global padded sequence length:    {max_seq_len} weeks")
    print(f"Total missing (param, week) cells flagged null: {total_missing_week_params:,}")
    print()
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()