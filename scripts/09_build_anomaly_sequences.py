"""
09_build_anomaly_sequences.py

HarvestSight -- Phase 5b: weekly weather ANOMALY sequences.

WHY THIS EXISTS
---------------
The 08 diagnostic showed At-Risk is predictable from weather ANOMALIES
(moisture especially, strongest in Kharif: single-feature test AUC ~0.58-0.59)
while the RNN trained on raw per-state-normalized weather scored test AUC
0.489. The raw input carries the monsoon/winter cycle, which dwarfs the
year-to-year deviations that actually move yield. This script removes the
cycle and leaves only "wetter / drier / hotter / cooler than usual for this
state, this season, this week".

WHAT IT DOES
------------
Reads rnn_sequences.json (the 10,261-record relabeled file) and writes a
file with the SAME record schema, but with every weekly weather value
replaced by an anomaly z-score:

    z = (x - clim_mean[state, season, week, param]) / scale[state, season, param]

  * Climatology is fitted on TRAIN-YEAR cells only (same temporal split as
    07/08: train 1997-2014), so val/test never touch it. No leakage.
  * Weather is shared by every crop in a (state, season, year) cell, so the
    climatology is built once per cell, not once per record.
  * The weekly climatological mean is lightly smoothed (3-week centered
    moving average). With ~18 train years a raw weekly mean is noisy
    (standard error ~0.24 sigma), and that noise would leak into every
    anomaly; smoothing cuts it to ~0.14 sigma while keeping the seasonal
    shape (monsoon onset, winter cooling). Smoothing is applied only
    where the full 3-week window fits: the first and last week of each
    season keep their raw means, because a one-sided window is biased where
    the climatology is steep (caught by a unit test on a synthetic cycle
    before release). Smoothing still slightly flattens sharp peaks (about
    2% of the amplitude for a 26-week cycle) -- a small, deliberate trade
    for lower noise.
  * The scale is the pooled std of the weekly deviations from that smoothed
    climatology over all train weeks of the (state, season, param) -- one
    stable number, not a per-week std. A per-week std from 18 samples is
    unreliable, and in dry weeks (rainfall ~0) it would turn small events
    into huge z-scores. By construction train anomalies have std ~1.
  * Anomalies are clipped to +/-5. The RNN trains with plain SGD and no
    gradient clipping; one extreme rain week must not blow up an update.
  * FULL 7-day weeks only: the final partial week (1 day in Kharif and in
    leap-year Rabi) is dropped. Its climatology would otherwise be built
    from a handful of leap-year cells. Season lengths become 26 / 26 / 17
    weeks (were 27 / 26-27 / 18) and `sequence_length` is updated to match.
  * Missing values stay None (07 already treats None as "at the normal").
  * Soil, crop, labels, yields: untouched, copied through.

ADDED FIELDS (07 ignores them; they exist for the next training script)
  season_onehot  [Kharif, Rabi, Summer] -- anomalies are season-relative, so
                 the model needs to be told the season explicitly.
  sequence_kind  "weekly_anomaly_z"

HOW TO USE THE RESULT WITH 07 UNCHANGED
---------------------------------------
Save/upload rnn_sequences_anomaly.json under the name rnn_sequences.json in
a SEPARATE folder/session from the original (do not overwrite the original).
07's schema, split (7,835 / 980 / 1,446) and training logic then run as-is;
its per-state normalization becomes a near no-op on anomalies (mean ~0,
std ~1). That isolates the input as the only change. The season indicator
needs a small change to 07's feature builder and is the next step.

OUTPUTS (written next to rnn_sequences.json)
  rnn_sequences_anomaly.json   compact JSON (no indentation) -- much smaller
                               than the 52 MB indented input
  weather_climatology.json     the fitted climatology + scales, which any
                               future inference script needs to turn new
                               weather into the same anomaly features

KNOWN LIMITS (stated, not hidden)
---------------------------------
  * Anomalies are measured against the TRAIN-period climatology; no climate
    trend is removed. The 08 diagnostic found temperature anomalies did not
    hold up out of sample, and a warming trend is the leading suspect. The
    per-split check printed below makes any drift visible (val/test mean far
    from 0). Trend adjustment is deliberately NOT done here, so its effect
    can be tested as a separate step.
  * Still ONE weather point per state.

Stdlib only. Runs in seconds.
"""

import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEATHER_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
SEASONS = ["Kharif", "Rabi", "Summer"]
SEASON_WINDOWS = {  # (start_month, start_day, end_month, end_day, crosses_into_next_year)
    "Kharif": (6, 1, 11, 30, False),
    "Rabi": (10, 1, 3, 31, True),
    "Summer": (3, 1, 6, 30, False),
}

N_TEST_YEARS = 3
TRAIN_FRACTION_OF_REMAINDER = 0.85   # identical to 07/08's split

MIN_CLIM_YEARS = 8        # train cells (and values per week) needed to fit a climatology
SMOOTH_HALF_WIDTH = 1     # 1 -> 3-week centered moving average
ANOMALY_CLIP = 5.0

# Benchmarks from the 08 run on the same file (a fresh run should match).
EXPECTED_RECORDS = 10261
EXPECTED_CELLS = 1377
EXPECTED_PAIRS = 62
EXPECTED_LENGTHS = {"Kharif": 26, "Rabi": 26, "Summer": 17}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _project_root():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:  # notebook cell (Colab): no __file__
        return None


def find_input(filename):
    root = _project_root()
    candidates = []
    if root is not None:
        candidates += [root / "cleaned_data" / filename, root / filename]
    candidates += [Path.cwd() / filename, Path("/content") / filename]
    for p in candidates:
        if p.exists():
            return p
    searched = "\n    ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Looked in:\n    {searched}")


# ---------------------------------------------------------------------------
# Split (identical to 07/08)
# ---------------------------------------------------------------------------

def temporal_split(records):
    years = sorted({r["year"] for r in records})
    test_years = set(years[-N_TEST_YEARS:])
    remainder_years = [y for y in years if y not in test_years]

    remainder_by_year = Counter(r["year"] for r in records if r["year"] in remainder_years)
    remainder_total = sum(remainder_by_year.values())

    train_years = set()
    cumulative = 0
    for y in remainder_years:  # oldest-first
        train_years.add(y)
        cumulative += remainder_by_year[y]
        if cumulative >= TRAIN_FRACTION_OF_REMAINDER * remainder_total:
            break
    val_years = set(remainder_years) - train_years
    return train_years, val_years, test_years


# ---------------------------------------------------------------------------
# Climatology and anomalies
# ---------------------------------------------------------------------------

def full_weeks_in_window(season, year):
    """Number of complete 7-day weeks in the season window for a crop-year."""
    sm, sd, em, ed, crosses = SEASON_WINDOWS[season]
    start = date(year, sm, sd)
    end = date(year + 1 if crosses else year, em, ed)
    return ((end - start).days + 1) // 7


def smooth(values, half_width=SMOOTH_HALF_WIDTH):
    """Centered moving average over the available (non-None) neighbors, applied
    only where the FULL window fits inside the series. Positions that are None
    stay None (gaps are not filled in), and the first and last `half_width`
    positions keep their raw value: a shrunken one-sided window is biased
    wherever the climatology is steep (e.g. monsoon onset at the start of
    Kharif) and would leave a systematic offset in exactly those weeks."""
    out = []
    n = len(values)
    for i in range(n):
        if values[i] is None:
            out.append(None)
            continue
        if i - half_width < 0 or i + half_width > n - 1:
            out.append(values[i])
            continue
        window = [v for v in values[i - half_width: i + half_width + 1] if v is not None]
        out.append(sum(window) / len(window))
    return out


def fit_climatology(cells, train_years):
    """(state, season) -> {"mean": [param][week], "scale": [param], "n_train_cells": int}
    from TRAIN-year cells only. Pairs with fewer than MIN_CLIM_YEARS train cells
    are omitted (their records are skipped downstream)."""
    by_pair = defaultdict(list)
    for (state, season, year), weeks in cells.items():
        if year in train_years:
            by_pair[(state, season)].append(weeks)

    n_params = len(WEATHER_PARAMS)
    clim = {}
    for pair, cell_list in by_pair.items():
        if len(cell_list) < MIN_CLIM_YEARS:
            continue
        n_weeks = max(len(cw) for cw in cell_list)

        mean_by_param = []
        for p in range(n_params):
            raw = []
            for w in range(n_weeks):
                vals = [cw[w][p] for cw in cell_list if len(cw) > w and cw[w][p] is not None]
                raw.append(sum(vals) / len(vals) if len(vals) >= MIN_CLIM_YEARS else None)
            mean_by_param.append(smooth(raw))

        scale = []
        for p in range(n_params):
            devs = []
            for cw in cell_list:
                for w in range(len(cw)):
                    x, m = cw[w][p], mean_by_param[p][w]
                    if x is not None and m is not None:
                        devs.append(x - m)
            if len(devs) >= 2:
                sd = math.sqrt(sum(d * d for d in devs) / len(devs))
                scale.append(sd if sd > 0 else 1.0)
            else:
                scale.append(1.0)

        clim[pair] = {"mean": mean_by_param, "scale": scale, "n_train_cells": len(cell_list)}
    return clim


def to_anomaly(weeks, entry, clip=ANOMALY_CLIP):
    """Weekly weather vectors -> anomaly z-scores. Returns (rows, n_clipped, n_values)."""
    rows = []
    n_clipped = 0
    n_values = 0
    for w, vec in enumerate(weeks):
        row = []
        for p, x in enumerate(vec):
            m = entry["mean"][p][w] if w < len(entry["mean"][p]) else None
            if x is None or m is None:
                row.append(None)
                continue
            z = (x - m) / entry["scale"][p]
            if z > clip:
                z, n_clipped = clip, n_clipped + 1
            elif z < -clip:
                z, n_clipped = -clip, n_clipped + 1
            n_values += 1
            row.append(round(z, 4))
        rows.append(row)
    return rows, n_clipped, n_values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seq_path = find_input("rnn_sequences.json")
    print(f"Loading sequence records from: {seq_path}")
    records = load_json(seq_path)
    print(f"  {len(records):,} records (benchmark {EXPECTED_RECORDS:,})")

    train_years, val_years, test_years = temporal_split(records)
    split_of_year = {}
    for y in train_years:
        split_of_year[y] = "train"
    for y in val_years:
        split_of_year[y] = "val"
    for y in test_years:
        split_of_year[y] = "test"
    print(f"Split -- train {min(train_years)}-{max(train_years)}  val {sorted(val_years)}  "
          f"test {sorted(test_years)}")
    if (min(train_years), max(train_years)) != (1997, 2014):
        print("  WARNING: train years differ from the 1997-2014 used in 07/08 -- is this the right file?")

    # ---- One weather cell per (state, season, year); full weeks only ----
    cells = {}
    for r in records:
        key = (r["state"], r["season"], r["year"])
        if key not in cells:
            n_use = min(r["sequence_length"], full_weeks_in_window(r["season"], r["year"]))
            cells[key] = [list(week) for week in r["sequence"][:n_use]]
    print(f"Weather cells (state, season, year): {len(cells):,}  (benchmark {EXPECTED_CELLS:,})")

    lengths = defaultdict(set)
    for (state, season, year), weeks in cells.items():
        lengths[season].add(len(weeks))
    for season in SEASONS:
        got = sorted(lengths[season])
        flag = "" if got == [EXPECTED_LENGTHS[season]] else f"   <-- benchmark {EXPECTED_LENGTHS[season]}"
        print(f"  {season:<7} full-week sequence length(s): {got}{flag}")

    # ---- Fit climatology on train years only ----
    clim = fit_climatology(cells, train_years)
    print(f"Climatology fitted for {len(clim)} (state, season) pairs (benchmark {EXPECTED_PAIRS}); "
          f"smoothing = {2 * SMOOTH_HALF_WIDTH + 1}-week moving average, clip = +/-{ANOMALY_CLIP}")

    # ---- Anomalies per cell ----
    anomalies = {}
    n_clipped_total = 0
    n_values_total = 0
    skipped_cells = 0
    for key, weeks in cells.items():
        state, season, _ = key
        entry = clim.get((state, season))
        if entry is None:
            anomalies[key] = None
            skipped_cells += 1
            continue
        rows, n_clipped, n_values = to_anomaly(weeks, entry)
        anomalies[key] = rows
        n_clipped_total += n_clipped
        n_values_total += n_values
    if skipped_cells:
        print(f"  WARNING: {skipped_cells} cells had no usable climatology; their records are skipped")
    print(f"Anomaly values: {n_values_total:,}; clipped at +/-{ANOMALY_CLIP}: {n_clipped_total:,} "
          f"({100.0 * n_clipped_total / max(1, n_values_total):.2f}%)")

    # ---- Sanity checks ----
    per_split_vals = {sp: [[] for _ in WEATHER_PARAMS] for sp in ("train", "val", "test")}
    for (state, season, year), rows in anomalies.items():
        if rows is None:
            continue
        sp = split_of_year[year]
        for row in rows:
            for p, z in enumerate(row):
                if z is not None:
                    per_split_vals[sp][p].append(z)

    def mean_std(vals):
        if not vals:
            return float("nan"), float("nan")
        m = sum(vals) / len(vals)
        return m, math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    print("\nAnomaly check by split, pooled over weather cells "
          "(train is ~0 mean / ~1 std by construction; val/test drift = trend or regime shift)")
    print(f"  {'param':<13}{'train mean/std':>18}{'val mean/std':>18}{'test mean/std':>18}")
    drift_flags = []
    for p, name in enumerate(WEATHER_PARAMS):
        cols = []
        for sp in ("train", "val", "test"):
            m, s = mean_std(per_split_vals[sp][p])
            cols.append(f"{m:+.3f} / {s:.3f}")
            if sp != "train" and abs(m) > 0.25:
                drift_flags.append(f"{name} {sp} mean {m:+.2f}")
        print(f"  {name:<13}{cols[0]:>18}{cols[1]:>18}{cols[2]:>18}")
    if drift_flags:
        print("  DRIFT: " + "; ".join(drift_flags) +
              "  (candidate for trend adjustment as a separate step)")
    else:
        print("  no val/test mean beyond +/-0.25 -- no obvious drift")

    # seasonal-cycle removal check: mean anomaly at each (season, week) position over train cells
    pos_sums = defaultdict(lambda: [[0.0, 0] for _ in WEATHER_PARAMS])
    for (state, season, year), rows in anomalies.items():
        if rows is None or year not in train_years:
            continue
        for w, row in enumerate(rows):
            for p, z in enumerate(row):
                if z is not None:
                    pos_sums[(season, w)][p][0] += z
                    pos_sums[(season, w)][p][1] += 1
    worst = [0.0] * len(WEATHER_PARAMS)
    for (season, w), per_param in pos_sums.items():
        for p, (s, n) in enumerate(per_param):
            if n:
                worst[p] = max(worst[p], abs(s / n))
    print("\nSeasonal-cycle removal check (train): largest |mean anomaly| at any (season, week) position")
    print("  " + "  ".join(f"{name}={worst[p]:.3f}" for p, name in enumerate(WEATHER_PARAMS)))
    print("  (near 0 = the seasonal cycle is gone; smoothing leaves a small residue, so expect values")
    print("   on the order of 0.1; a value near or above ~0.5 would mean the cycle is NOT being removed)")

    # ---- Write records ----
    max_len = max(len(weeks) for weeks in cells.values())
    zero_week = [0.0] * len(WEATHER_PARAMS)
    out_records = []
    skipped_records = 0
    for r in records:
        key = (r["state"], r["season"], r["year"])
        rows = anomalies.get(key)
        if rows is None:
            skipped_records += 1
            continue
        out = dict(r)
        out["sequence"] = [list(row) for row in rows] + [list(zero_week) for _ in range(max_len - len(rows))]
        out["sequence_length"] = len(rows)
        out["season_onehot"] = [1.0 if r["season"] == s else 0.0 for s in SEASONS]
        out["sequence_kind"] = "weekly_anomaly_z"
        out_records.append(out)

    out_path = seq_path.parent / "rnn_sequences_anomaly.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, separators=(",", ":"))

    clim_out = {}
    for (state, season), entry in clim.items():
        clim_out.setdefault(state, {})[season] = {
            "mean": {name: entry["mean"][p] for p, name in enumerate(WEATHER_PARAMS)},
            "scale": {name: entry["scale"][p] for p, name in enumerate(WEATHER_PARAMS)},
            "n_train_cells": entry["n_train_cells"],
        }
    clim_path = seq_path.parent / "weather_climatology.json"
    with open(clim_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "train_years": sorted(train_years),
                "weather_params": WEATHER_PARAMS,
                "smooth_window_weeks": 2 * SMOOTH_HALF_WIDTH + 1,
                "anomaly_clip": ANOMALY_CLIP,
                "min_clim_years": MIN_CLIM_YEARS,
                "full_weeks_only": True,
                "formula": "z = (x - mean[state][season][param][week]) / scale[state][season][param]",
            },
            "climatology": clim_out,
        }, f, separators=(",", ":"))

    print()
    print("=" * 66)
    print(f"Records written: {len(out_records):,}  (skipped {skipped_records:,}; benchmark {EXPECTED_RECORDS:,} written, 0 skipped)")
    split_counts = Counter(split_of_year[r["year"]] for r in out_records)
    print(f"  train={split_counts['train']:,}  val={split_counts['val']:,}  test={split_counts['test']:,}"
          f"   (07 expects 7,835 / 980 / 1,446)")
    print(f"Padded length: {max_len} weeks; real lengths per season shown above")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {clim_path}")
    print("\nNEXT: to run 07 unchanged, upload rnn_sequences_anomaly.json as rnn_sequences.json in a")
    print("SEPARATE session/folder from the original. (On Colab, download both outputs before the runtime resets.)")


if __name__ == "__main__":
    main()