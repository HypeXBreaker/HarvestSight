"""
05_derive_risk_labels.py   (v2 -- detrended residual labels)

HarvestSight -- Phase 4: Target Definition (Risk Labeling).

WHY THIS VERSION EXISTS (v1 -> v2)
----------------------------------
v1 scored every year against its (state, crop, season) group's mean/std over
ALL years. Yields trend upward over 1997-2019, so early years sat below that
long-run mean (labeled At-Risk) and recent years sat above it. Measured on
the real label file, the At-Risk share fell steadily with time:

    1997-02: 59.4%   2003-08: 40.1%   2009-14: 18.9%   val 2015-16: 18.1%   test 2017-19: 10.8%

The label was partly encoding WHICH ERA a year belongs to -- something
weather cannot explain -- and the model trained on a 38.5% At-Risk base
rate was then tested on a 10.8% one (its At-Risk precision/recall on test
were indistinguishable from chance at the same flag rate).

v2 REMOVES THE ERA EFFECT by detrending. For each (state, crop, season)
group, fit an ordinary-least-squares line  yield = a + b * year  across
the group's years, then score every year by how far it sits from ITS OWN
trend-expected yield:

    residual = yield - (a + b * year)
    z        = residual / residual_std          (residual_std uses n - 2 dof)

    z >= -0.5            -> Low
    z <  -0.5            -> At-Risk              (primary label, feeds the model)
    -1.0 <= z < -0.5     -> Moderate  \\  reference-only `risk_detail`
    z <  -1.0            -> High      /  (kept for traceability, not fed to the model)

Same -0.5 / -1.0 cutoffs as v1; only the baseline changed (a fitted trend
instead of a flat all-years mean).

CHOSEN OVER THE TRAILING-WINDOW ALTERNATIVE (validated on the real yield data
before writing this -- At-Risk share by period, same 21 crops):

    method                        labeled   97-02  03-08  09-14  val    test
    v1 all-years mean              10,664   .594   .401   .189   .181   .108
    detrended, min 8 yrs  (THIS)   10,261   .258   .318   .278   .318   .230
    trailing window W=7             7,689   .450   .233   .178   .235   .178
    trailing window W=10            7,689   .450   .227   .151   .202   .138

The trailing window loses ~28% of records (the first years of every group
have no history) and still drifts, because a trailing mean lags a rising
trend. Detrending keeps almost all records and is flat to within a few
points across eras.

CHANGES VS v1
-------------
  * Baseline: flat all-years mean  ->  per-group linear trend.
  * MIN_YEARS: 5 -> 8. A 2-parameter line fit on 5-7 points is unstable;
    groups below 8 years get `insufficient_data` (no label, excluded
    downstream). Expect ~675 insufficient_data records (v1 had 272).
  * Output schema is UNCHANGED (state, crop, season, year, yield,
    baseline_mean, baseline_std, z, risk, risk_detail), so
    06_build_weekly_sequences.py needs NO code changes -- just re-run it.
    Field meaning shifts slightly: `baseline_mean` is now the trend-expected
    yield FOR THAT YEAR, and `baseline_std` is the residual std.
    New fields added for traceability: `trend_slope_per_year`, `baseline_years`.
  * Console summary now prints the At-Risk share by period and by year,
    so any remaining era drift is visible immediately after the run.

KNOWN CAVEATS (stated, not hidden)
----------------------------------
  * The trend is fit on every year in the group, including the val/test
    years. Labels for earlier years therefore depend slightly on later
    yields. This is standard practice for yield-anomaly detrending (fitting
    on train years only and extrapolating would push accelerating recent
    years above the line and reintroduce the drift). It affects labels,
    not model inputs.
  * A straight line can under-fit a trend that accelerates late in the
    period; the expected test-period share (23.0%) sits a few points below
    the earlier eras (26-32%) for that reason. Much smaller than v1's
    59% -> 11% slide, but not zero. A curved trend is the next step if the
    model results suggest it matters.
  * Zero-variance groups (residual std == 0) default to Low with z left null,
    as in v1.

Input:
  cleaned_data/state_crop_season_year_yield.json   (Phase 2 output)
  cleaned_data/crop_shortlist.json                 (Phase 3 output, 21 crops)

Output:
  cleaned_data/yield_with_risk_labels.json

Stdlib only.
"""

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"

YIELD_INPUT_PATH = CLEANED_DATA_DIR / "state_crop_season_year_yield.json"
SHORTLIST_INPUT_PATH = CLEANED_DATA_DIR / "crop_shortlist.json"
OUTPUT_PATH = CLEANED_DATA_DIR / "yield_with_risk_labels.json"

MIN_YEARS = 8          # groups with fewer years -> insufficient_data, no label
AT_RISK_Z = -0.5       # z below this -> At-Risk (primary label)
HIGH_Z = -1.0          # z below this -> High (reference label only)
ZERO_STD_EPS = 1e-12   # residual std at/below this counts as zero variance

# (name, first_year, last_year) -- the last three match the 07 training split.
PERIODS = [
    ("1997-2002", 1997, 2002),
    ("2003-2008", 2003, 2008),
    ("2009-2014", 2009, 2014),
    ("val 2015-16", 2015, 2016),
    ("test 2017-19", 2017, 2019),
]

# Benchmark from the pre-build validation on the real data (same method,
# MIN_YEARS=8). A fresh run should land close to these.
BENCHMARK_LABELED = 10261
BENCHMARK_SHARES = [0.258, 0.318, 0.278, 0.318, 0.230]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_risk(z):
    """Primary 2-class label, fed to the model."""
    return "Low" if z >= AT_RISK_Z else "At-Risk"


def classify_risk_detail(z):
    """Reference-only 3-class label, kept for traceability."""
    if z >= AT_RISK_Z:
        return "Low"
    if z >= HIGH_Z:
        return "Moderate"
    return "High"


def fit_linear_trend(points):
    """
    Ordinary least squares of yield on year for one group.

    points: list of (year, yield), already sorted by year, len >= MIN_YEARS.
    Returns (slope, fitted_values, residuals, residual_std), with
    residual_std computed on n - 2 degrees of freedom (two fitted params).
    """
    n = len(points)
    xs = [float(year) for year, _ in points]
    ys = [value for _, value in points]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:  # cannot happen with unique years; guard anyway -> flat line
        slope = 0.0
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    residuals = [y - f for y, f in zip(ys, fitted)]
    sse = sum(e * e for e in residuals)
    residual_std = math.sqrt(sse / (n - 2))
    return slope, fitted, residuals, residual_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading yield records from: {YIELD_INPUT_PATH}")
    yield_records = load_json(YIELD_INPUT_PATH)
    if isinstance(yield_records, dict):  # tolerate a {"records": [...]} wrapper
        yield_records = yield_records["records"]
    print(f"  {len(yield_records):,} total (state, crop, season, year) cells (Phase 2 output)")

    print(f"Loading crop shortlist from: {SHORTLIST_INPUT_PATH}")
    crop_shortlist_raw = load_json(SHORTLIST_INPUT_PATH)
    crop_shortlist = crop_shortlist_raw["crops"] if isinstance(crop_shortlist_raw, dict) else crop_shortlist_raw
    shortlist_set = set(crop_shortlist)
    print(f"  {len(shortlist_set)} shortlisted crops")

    filtered = [r for r in yield_records if r["crop"] in shortlist_set]
    print(
        f"Filtered to shortlisted crops: {len(filtered):,} records "
        f"(dropped {len(yield_records) - len(filtered):,} non-shortlisted-crop records)"
    )

    seen_crops = {r["crop"] for r in filtered}
    missing_crops = shortlist_set - seen_crops
    if missing_crops:
        print(
            f"  WARNING: {len(missing_crops)} shortlisted crop(s) matched zero records "
            f"-- possible name mismatch: {sorted(missing_crops)}"
        )

    # Group by (state, crop, season)
    groups = {}
    for r in filtered:
        groups.setdefault((r["state"], r["crop"], r["season"]), []).append(r)
    print(f"Grouped into {len(groups):,} (state, crop, season) groups")

    output_records = []
    insufficient_groups = 0
    zero_std_groups = 0
    risk_counts = {"Low": 0, "At-Risk": 0, "insufficient_data": 0}
    risk_detail_counts = {"Low": 0, "Moderate": 0, "High": 0, "insufficient_data": 0}

    for (state, crop, season), records in groups.items():
        records = sorted(records, key=lambda r: r["year"])
        n_years = len(records)

        # Coverage guard
        if n_years < MIN_YEARS:
            insufficient_groups += 1
            for r in records:
                output_records.append({
                    "state": state, "crop": crop, "season": season,
                    "year": r["year"], "yield": r["yield"],
                    "baseline_mean": None, "baseline_std": None, "z": None,
                    "trend_slope_per_year": None, "baseline_years": n_years,
                    "risk": "insufficient_data", "risk_detail": "insufficient_data",
                })
                risk_counts["insufficient_data"] += 1
                risk_detail_counts["insufficient_data"] += 1
            continue

        # Per-group linear trend, then residual z-scores
        points = [(r["year"], r["yield"]) for r in records]
        slope, fitted, residuals, residual_std = fit_linear_trend(points)

        # Zero-variance guard
        if residual_std <= ZERO_STD_EPS:
            zero_std_groups += 1
            for r, f in zip(records, fitted):
                output_records.append({
                    "state": state, "crop": crop, "season": season,
                    "year": r["year"], "yield": r["yield"],
                    "baseline_mean": f, "baseline_std": residual_std, "z": None,
                    "trend_slope_per_year": slope, "baseline_years": n_years,
                    "risk": "Low", "risk_detail": "Low",
                })
                risk_counts["Low"] += 1
                risk_detail_counts["Low"] += 1
            continue

        for r, f, e in zip(records, fitted, residuals):
            z = e / residual_std
            risk = classify_risk(z)
            risk_detail = classify_risk_detail(z)
            output_records.append({
                "state": state, "crop": crop, "season": season,
                "year": r["year"], "yield": r["yield"],
                "baseline_mean": f, "baseline_std": residual_std, "z": z,
                "trend_slope_per_year": slope, "baseline_years": n_years,
                "risk": risk, "risk_detail": risk_detail,
            })
            risk_counts[risk] += 1
            risk_detail_counts[risk_detail] += 1

    output_records.sort(key=lambda r: (r["state"], r["crop"], r["season"], str(r["year"])))

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    total = len(output_records)
    labeled = [r for r in output_records if r["risk"] != "insufficient_data"]
    print()
    print("=" * 66)
    print("Phase 4 -- Risk Labeling (v2, detrended): Summary")
    print("=" * 66)
    print(f"Total records processed:   {total:,}")
    print(
        f"Groups below MIN_YEARS={MIN_YEARS}:  {insufficient_groups:,} groups "
        f"-> {risk_counts['insufficient_data']:,} records"
    )
    print(f"Zero-variance guard triggered:  {zero_std_groups:,} groups -> defaulted to Low")
    print()
    print("Primary label (risk) breakdown -- feeds the model:")
    for label in ("Low", "At-Risk", "insufficient_data"):
        count = risk_counts[label]
        pct = (count / total * 100) if total else 0.0
        print(f"  {label:<18} {count:>7,}  ({pct:5.1f}%)")
    print()
    print("Reference label (risk_detail) breakdown -- traceability only:")
    for label in ("Low", "Moderate", "High", "insufficient_data"):
        count = risk_detail_counts[label]
        pct = (count / total * 100) if total else 0.0
        print(f"  {label:<18} {count:>7,}  ({pct:5.1f}%)")

    # --- Era-drift check: At-Risk share by period (flat = drift removed) ---
    print()
    print("At-Risk share by period (the drift check -- v1 slid 59% -> 11%):")
    print(f"  {'period':<14}{'labeled':>9}{'At-Risk':>9}{'share':>8}{'benchmark':>11}")
    shares = []
    for (name, lo, hi), bench in zip(PERIODS, BENCHMARK_SHARES):
        in_period = [r for r in labeled if lo <= r["year"] <= hi]
        n = len(in_period)
        a = sum(1 for r in in_period if r["risk"] == "At-Risk")
        share = (a / n) if n else float("nan")
        shares.append(share)
        print(f"  {name:<14}{n:>9,}{a:>9,}{share:>8.3f}{bench:>11.3f}")
    valid = [s for s in shares if s == s]
    if valid:
        spread = max(valid) - min(valid)
        verdict = "OK -- era effect largely removed" if spread <= 0.15 else "WARNING -- still drifting, investigate"
        print(f"  spread (max - min across periods): {spread:.3f}  [{verdict}]")
    print(f"  labeled records: {len(labeled):,}  (benchmark {BENCHMARK_LABELED:,})")

    print()
    print("At-Risk share by year:")
    years = sorted({r["year"] for r in labeled})
    for y in years:
        yr = [r for r in labeled if r["year"] == y]
        a = sum(1 for r in yr if r["risk"] == "At-Risk")
        print(f"  {y}  n={len(yr):>4}  At-Risk={a / len(yr):.3f}")

    print()
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()