"""
13_kharif_suitability.py

HarvestSight -- Option B: turn the Kharif moisture index into something a
person can read and use: three RISK TIERS, and a per-state sensitivity table.
NO TRAINING. Nothing is fitted to labels; the only fitted quantities are the
anomaly climatology (train years, as in 08/11) and the tier cut points
(train-year terciles of the index).

THE INDEX (unchanged from 11)
-----------------------------
risk score = - mean( anomaly z of PRECTOTCORR_season_mean, GWETROOT_season_mean,
RH2M_season_mean ), anomalies against the (state, season) climatology fitted on
TRAIN years only. Wetter than usual => lower score => lower risk. Because it
depends only on the state-season-year weather, every crop in the same
(state, Kharif, year) shares one score.

PART 1 -- RISK TIERS (the deliverable)
--------------------------------------
Cut points = the 1/3 and 2/3 quantiles of the score over Kharif TRAIN rows.
  Low risk (wet)    score <= lower cut
  Medium            between
  High risk (dry)   score >  upper cut
For each split (train / val / test) the script reports, per tier: the share of
records, the observed At-Risk rate, and a (state, year) cluster-bootstrap 95%
CI. A plain-language rule falls out: "a Kharif season that is drier than the
state's usual (index above the upper cut) went with At-Risk crops X% of the
time versus Y% in wet seasons."

PRE-REGISTERED READING OF THE TIERS (fixed before running; judged on TEST)
-------------------------------------------------------------------------
  PASS     At-Risk rate rises strictly Low < Medium < High, AND the
           High-minus-Low difference has cluster-CI lower bound > 0
  FAIL     High-tier rate <= Low-tier rate (no separation at all)
  PARTIAL  anything else (e.g. non-monotone middle, or CI includes 0)

PART 2 -- PER-STATE SENSITIVITY (exploratory, NO verdict)
---------------------------------------------------------
For each state with enough Kharif data (>= 10 years, >= 15 At-Risk and >= 15
Low records): all-years AUC of the index with a year-cluster bootstrap CI.
  SENSITIVE   AUC >= 0.56 and CI lower bound > 0.50
  REVERSED    AUC <= 0.44 and CI upper bound < 0.50
  -           otherwise
Also printed: how many states have AUC > 0.5 with a one-sided sign test, and
how many SENSITIVE flags chance alone would produce. A state with a flat or
unclear AUC means "this index does not tell us much here" -- NOT "weather does
not matter there". The test years are only 3 per state, so nothing per-state
can be confirmed out of sample; this is a map of where the rule works in the
historical record.

OUTPUTS
-------
  kharif_suitability.json (next to the input): cut points, tier tables per
  split, tier verdict, per-state table, and a (state, year) table of index
  score, tier, and observed At-Risk share -- ready for a later display layer.

REPRODUCTION CHECKS (printed first)
-----------------------------------
  * Kharif test AUCs of the three features must equal 08/11 (0.594/0.586/0.581).
  * Pooled Kharif all-years AUC of the index must equal 11's 0.608.

CAVEATS
-------
  * Tier cut points use train years; later years are drier or wetter than
    the train climatology (09 found test soil wetness +0.65 sigma), so test
    tier SHARES will not be 1/3 each. Rates within tiers are the point.
  * The index's feature choice came from 08, which saw all splits; the test
    tier table is therefore not a clean out-of-sample confirmation of the
    feature choice (the cut points and tier definitions are fixed on train).
  * Several tests across states: expect a few flags by chance.
  * One weather point per state; state-aggregated yields.

Input:
  rnn_sequences.json   (the RAW relabeled sequence file, not the anomaly one)
Output:
  kharif_suitability.json

Runtime: seconds. Stdlib only.
"""

import json
import math
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEATHER_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
SEGMENTS = ("season", "early", "mid", "late")
FEATURE_NAMES = [f"{p}_{seg}_mean" for p in WEATHER_PARAMS for seg in SEGMENTS] + ["PRECTOTCORR_max_week"]

SEASON_WINDOWS = {  # (start_month, start_day, end_month, end_day, crosses_into_next_year)
    "Kharif": (6, 1, 11, 30, False),
    "Rabi": (10, 1, 3, 31, True),
    "Summer": (3, 1, 6, 30, False),
}
SCOPE_SEASON = "Kharif"

N_TEST_YEARS = 3
TRAIN_FRACTION_OF_REMAINDER = 0.85   # identical to 07/08/11
MIN_CLIM_YEARS = 8
BOOTSTRAP_B = 500
RANDOM_SEED = 42

MOISTURE_FEATURES = ["PRECTOTCORR_season_mean", "GWETROOT_season_mean", "RH2M_season_mean"]

TIERS = ["Low", "Medium", "High"]
TIER_DISPLAY = {"Low": "Low risk (wet)", "Medium": "Medium", "High": "High risk (dry)"}

# per-state eligibility and flags
STATE_MIN_YEARS = 10
STATE_MIN_CLASS = 15
SIGNAL_AUC = 0.56
REVERSED_AUC = 0.44
FALSE_POSITIVE_RATE = 0.025

EXPECTED_RECORDS = 10261
EXPECTED_CELLS = 1377
EXPECTED_KHARIF_RECORDS = 5445
BENCH_08_KHARIF_TEST = {"PRECTOTCORR_season_mean": 0.594, "GWETROOT_season_mean": 0.586,
                        "RH2M_season_mean": 0.581}
BENCH_11_KHARIF_ALLYEARS_AUC = 0.608


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
# Split, features, climatology (identical to 08 / 11)
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


def full_weeks_in_window(season, year):
    """Number of complete 7-day weeks in the season window for a crop-year."""
    sm, sd, em, ed, crosses = SEASON_WINDOWS[season]
    start = date(year, sm, sd)
    end = date(year + 1 if crosses else year, em, ed)
    return ((end - start).days + 1) // 7


def season_features(record):
    """25 season-level weather features from one record's real, full weeks."""
    n_use = min(record["sequence_length"], full_weeks_in_window(record["season"], record["year"]))
    weeks = record["sequence"][:n_use]
    a, b = n_use // 3, (2 * n_use) // 3
    segments = {"season": weeks, "early": weeks[:a], "mid": weeks[a:b], "late": weeks[b:]}

    feats = {}
    for i, p in enumerate(WEATHER_PARAMS):
        for seg_name, seg in segments.items():
            vals = [w[i] for w in seg if w[i] is not None]
            feats[f"{p}_{seg_name}_mean"] = (sum(vals) / len(vals)) if vals else None
    rain = [w[0] for w in weeks if w[0] is not None]  # index 0 = PRECTOTCORR
    feats["PRECTOTCORR_max_week"] = max(rain) if rain else None
    return feats


def build_climatology(cells, train_years):
    """(state, season) -> feature -> (mean, std or None), from TRAIN-year cells only."""
    buckets = defaultdict(lambda: defaultdict(list))
    for (state, season, year), feats in cells.items():
        if year in train_years:
            for f, v in feats.items():
                if v is not None:
                    buckets[(state, season)][f].append(v)
    clim = {}
    for key, fmap in buckets.items():
        clim[key] = {}
        for f, vals in fmap.items():
            if len(vals) >= MIN_CLIM_YEARS:
                m = sum(vals) / len(vals)
                var = sum((v - m) ** 2 for v in vals) / len(vals)
                clim[key][f] = (m, math.sqrt(var) if var > 0 else None)
    return clim


def moisture_index(anomalies):
    """Mean of the available moisture anomalies (higher = wetter than usual)."""
    vals = [anomalies[f] for f in MOISTURE_FEATURES if anomalies.get(f) is not None]
    return (sum(vals) / len(vals)) if vals else None


# ---------------------------------------------------------------------------
# Metrics (identical to 08 / 11)
# ---------------------------------------------------------------------------

def roc_auc(labels, scores, positive=1):
    """Rank-based ROC-AUC (Mann-Whitney U), ties counting half. None if a class is absent."""
    n_pos = sum(1 for t in labels if t == positive)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == positive:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ci95(vals):
    vals = sorted(vals)
    last = len(vals) - 1
    return vals[int(0.025 * last)], vals[int(0.975 * last)]


def cluster_bootstrap_ci(items, rng, b=None):
    """95% CI for AUC, resampling whole clusters.
    items: list of (cluster_id, label, score). Returns (lo, hi) or (None, None)."""
    b = BOOTSTRAP_B if b is None else b
    clusters = defaultdict(list)
    for cid, y, s in items:
        clusters[cid].append((y, s))
    ids = list(clusters)
    if len(ids) < 5:
        return None, None
    aucs = []
    for _ in range(b):
        labels, scores = [], []
        for _ in range(len(ids)):
            for y, s in clusters[ids[rng.randrange(len(ids))]]:
                labels.append(y)
                scores.append(s)
        a = roc_auc(labels, scores, 1)
        if a is not None:
            aucs.append(a)
    if len(aucs) < 20:
        return None, None
    return _ci95(aucs)


# ---------------------------------------------------------------------------
# New in 13: tiers
# ---------------------------------------------------------------------------

def tercile_cuts(train_scores):
    """Lower and upper cut points (1/3 and 2/3 quantiles) of the train scores."""
    s = sorted(train_scores)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def tier_of(score, cuts):
    lo, hi = cuts
    if score <= lo:
        return "Low"
    if score <= hi:
        return "Medium"
    return "High"


def tier_bootstrap(rows, rng, b=None):
    """(state, year)-cluster bootstrap 95% CIs for the At-Risk rate of each tier
    and for the High-minus-Low difference. Resamples whole clusters."""
    b = BOOTSTRAP_B if b is None else b
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["cluster"]].append((r["tier"], r["y"]))
    ids = list(clusters)
    if len(ids) < 5:
        return None
    samples = {t: [] for t in TIERS}
    diffs = []
    for _ in range(b):
        n = {t: 0 for t in TIERS}
        pos = {t: 0 for t in TIERS}
        for _ in range(len(ids)):
            for tier, y in clusters[ids[rng.randrange(len(ids))]]:
                n[tier] += 1
                pos[tier] += y
        rates = {t: (pos[t] / n[t] if n[t] else None) for t in TIERS}
        for t in TIERS:
            if rates[t] is not None:
                samples[t].append(rates[t])
        if rates["High"] is not None and rates["Low"] is not None:
            diffs.append(rates["High"] - rates["Low"])
    out = {"rates": {t: (_ci95(samples[t]) if len(samples[t]) >= 20 else None) for t in TIERS},
           "diff": _ci95(diffs) if len(diffs) >= 20 else None}
    return out


def tier_report(rows, rng):
    """Per-tier counts, shares, At-Risk rates and CIs for one set of rows."""
    n = len(rows)
    base = (sum(r["y"] for r in rows) / n) if n else None
    boot = tier_bootstrap(rows, rng) if n else None
    tiers = {}
    for t in TIERS:
        sub = [r for r in rows if r["tier"] == t]
        pos = sum(r["y"] for r in sub)
        rate = (pos / len(sub)) if sub else None
        tiers[t] = {"n": len(sub), "n_pos": pos, "share": (len(sub) / n) if n else None,
                    "rate": rate, "ci95": boot["rates"][t] if boot else None,
                    "lift_vs_base": (rate / base) if (rate is not None and base) else None}
    rates = [tiers[t]["rate"] for t in TIERS]
    monotone = all(r is not None for r in rates) and rates[0] < rates[1] < rates[2]
    diff = (tiers["High"]["rate"] - tiers["Low"]["rate"]
            if tiers["High"]["rate"] is not None and tiers["Low"]["rate"] is not None else None)
    return {"n": n, "base_rate": base, "tiers": tiers, "monotone": monotone,
            "diff_high_low": diff, "diff_ci95": boot["diff"] if boot else None}


def tier_verdict(report):
    """Pre-registered reading (see module docstring)."""
    diff, ci = report["diff_high_low"], report["diff_ci95"]
    if diff is None or diff <= 0:
        return "FAIL"
    if report["monotone"] and ci is not None and ci[0] > 0:
        return "PASS"
    return "PARTIAL"


# ---------------------------------------------------------------------------
# New in 13: per-state sensitivity
# ---------------------------------------------------------------------------

def state_stats(rows, rng):
    ys = [r["y"] for r in rows]
    n_pos = sum(ys)
    n_years = len({r["year"] for r in rows})
    out = {"n": len(rows), "n_pos": n_pos, "n_years": n_years,
           "base_rate": (n_pos / len(rows)) if rows else None}
    out["eligible"] = (n_years >= STATE_MIN_YEARS and n_pos >= STATE_MIN_CLASS
                       and (len(rows) - n_pos) >= STATE_MIN_CLASS)
    out["auc"] = roc_auc(ys, [r["score"] for r in rows], 1) if rows else None
    out["ci_lo"] = out["ci_hi"] = None
    if out["eligible"] and out["auc"] is not None:
        # records in the same (state, year) share weather -> cluster by year within a state
        out["ci_lo"], out["ci_hi"] = cluster_bootstrap_ci(
            [(r["year"], r["y"], r["score"]) for r in rows], rng)
    auc, lo, hi = out["auc"], out["ci_lo"], out["ci_hi"]
    out["sensitive"] = bool(out["eligible"] and auc is not None and lo is not None
                            and auc >= SIGNAL_AUC and lo > 0.5)
    out["reversed"] = bool(out["eligible"] and auc is not None and hi is not None
                           and auc <= REVERSED_AUC and hi < 0.5)
    out["flag"] = "SENSITIVE" if out["sensitive"] else ("REVERSED" if out["reversed"] else "-")
    return out


def sign_test_one_sided(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return None
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def fmt_auc(auc, lo=None, hi=None):
    if auc is None:
        return "   --  "
    s = f"{auc:.3f}"
    if lo is not None and hi is not None:
        s += f" [{lo:.3f},{hi:.3f}]"
    return s


def fmt_rate(rate, ci):
    if rate is None:
        return "   --"
    s = f"{rate:.3f}"
    if ci is not None:
        s += f" [{ci[0]:.3f},{ci[1]:.3f}]"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    seq_path = find_input("rnn_sequences.json")
    print(f"Loading sequence records from: {seq_path}")
    records = load_json(seq_path)
    kind = records[0].get("sequence_kind", "raw_weekly_weather") if records else "unknown"
    if kind == "weekly_anomaly_z":
        raise ValueError("This is the weekly-ANOMALY file. 13 (like 08/11) needs the RAW relabeled "
                         "rnn_sequences.json; upload that file instead.")
    print(f"  {len(records):,} records (benchmark {EXPECTED_RECORDS:,})  |  input kind: {kind}")

    train_years, val_years, test_years = temporal_split(records)
    split_of_year = {y: "train" for y in train_years}
    split_of_year.update({y: "val" for y in val_years})
    split_of_year.update({y: "test" for y in test_years})
    print(f"Split -- train {min(train_years)}-{max(train_years)}  val {sorted(val_years)}  test {sorted(test_years)}")

    cells = {}
    for r in records:
        key = (r["state"], r["season"], r["year"])
        if key not in cells:
            cells[key] = season_features(r)
    print(f"Weather cells (state, season, year): {len(cells):,} (benchmark {EXPECTED_CELLS:,})")

    clim = build_climatology(cells, train_years)
    anomalies = {}
    for key, feats in cells.items():
        state, season, _ = key
        stats = clim.get((state, season))
        if not stats:
            anomalies[key] = None
            continue
        an = {}
        for f in FEATURE_NAMES:
            v, st = feats.get(f), stats.get(f)
            if v is None or st is None:
                an[f] = None
            else:
                m, sd = st
                an[f] = 0.0 if sd is None else (v - m) / sd
        anomalies[key] = an

    rows_all = []
    for r in records:
        an = anomalies.get((r["state"], r["season"], r["year"]))
        idx = moisture_index(an) if an is not None else None
        if idx is None:
            continue
        rows_all.append({"state": r["state"], "crop": r["crop"], "season": r["season"], "year": r["year"],
                         "split": split_of_year[r["year"]], "cluster": (r["state"], r["year"]),
                         "y": 1 if r["risk"] == "At-Risk" else 0, "score": -idx, "an": an})
    rows = [r for r in rows_all if r["season"] == SCOPE_SEASON]
    print(f"Rows scored: {len(rows_all):,} of {len(records):,}  |  {SCOPE_SEASON} rows: {len(rows):,} "
          f"(benchmark {EXPECTED_KHARIF_RECORDS:,})")

    # ---- 0. Reproduction checks ----
    print("\n" + "=" * 78)
    print("0. REPRODUCTION CHECKS (Kharif)")
    print("=" * 78)
    kh_test = [r for r in rows if r["split"] == "test"]
    all_match = True
    for f in MOISTURE_FEATURES:
        sub = [(r["y"], -r["an"][f]) for r in kh_test if r["an"].get(f) is not None]
        a = roc_auc([y for y, _ in sub], [s for _, s in sub], 1)
        ok = a is not None and abs(a - BENCH_08_KHARIF_TEST[f]) < 0.002
        all_match = all_match and ok
        print(f"  test AUC {f:<26} {a:.3f}   08/11 reported {BENCH_08_KHARIF_TEST[f]:.3f}   "
              f"{'MATCH' if ok else 'DIFFERS'}")
    pooled_auc = roc_auc([r["y"] for r in rows], [r["score"] for r in rows], 1)
    ok = pooled_auc is not None and abs(pooled_auc - BENCH_11_KHARIF_ALLYEARS_AUC) < 0.002
    all_match = all_match and ok
    print(f"  Kharif all-years index AUC       {pooled_auc:.3f}   11 reported {BENCH_11_KHARIF_ALLYEARS_AUC:.3f}   "
          f"{'MATCH' if ok else 'DIFFERS'}")
    print("  " + ("Pipeline reproduces 08 and 11 exactly." if all_match else
                  "WARNING: does not reproduce 08/11 -- check the input file before trusting anything below."))

    # ---- 1. Tiers ----
    train_scores = [r["score"] for r in rows if r["split"] == "train"]
    cuts = tercile_cuts(train_scores)
    for r in rows:
        r["tier"] = tier_of(r["score"], cuts)

    print("\n" + "=" * 78)
    print("1. KHARIF RISK TIERS (cut points = train-year terciles of the index)")
    print("=" * 78)
    print(f"  index = -(mean anomaly z of rainfall, root-zone wetness, humidity); "
          f"higher = drier than usual")
    print(f"  lower cut {cuts[0]:+.3f}   upper cut {cuts[1]:+.3f}   (from {len(train_scores):,} train rows)")

    split_reports = {}
    for sp in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == sp]
        split_reports[sp] = tier_report(sub, rng)

    for sp in ("train", "val", "test"):
        rep = split_reports[sp]
        print(f"\n  {sp.upper()}  ({rep['n']:,} rows, overall At-Risk rate {rep['base_rate']:.3f})")
        print(f"    {'tier':<18}{'n':>6}{'share':>8}{'At-Risk':>9}   {'At-Risk rate [cluster 95% CI]':<32}{'lift':>7}")
        for t in TIERS:
            d = rep["tiers"][t]
            lift = f"x{d['lift_vs_base']:.2f}" if d["lift_vs_base"] is not None else "  --"
            print(f"    {TIER_DISPLAY[t]:<18}{d['n']:>6,}{d['share']:>8.2f}{d['n_pos']:>9,}   "
                  f"{fmt_rate(d['rate'], d['ci95']):<32}{lift:>7}")
        dci = rep["diff_ci95"]
        dtxt = (f"{rep['diff_high_low']:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]"
                if rep["diff_high_low"] is not None and dci is not None else "--")
        print(f"    High minus Low: {dtxt}   monotone Low<Medium<High: {'yes' if rep['monotone'] else 'no'}")

    verdict = tier_verdict(split_reports["test"])
    print("\n  TIER VERDICT on TEST (pre-registered):")
    print("    PASS     strictly monotone Low<Medium<High and High-minus-Low CI lower bound > 0")
    print("    FAIL     High rate <= Low rate")
    print("    PARTIAL  otherwise")
    print(f"    --> {verdict}")
    test_shares = {t: split_reports["test"]["tiers"][t]["share"] for t in TIERS}
    train_shares = {t: split_reports["train"]["tiers"][t]["share"] for t in TIERS}
    print(f"    (tier shares train {train_shares['Low']:.2f}/{train_shares['Medium']:.2f}/{train_shares['High']:.2f}"
          f" vs test {test_shares['Low']:.2f}/{test_shares['Medium']:.2f}/{test_shares['High']:.2f}"
          f" -- a shift means the test years' moisture differs from the train climatology)")

    # ---- 2. Per-state sensitivity ----
    by_state = defaultdict(list)
    for r in rows:
        by_state[r["state"]].append(r)
    state_results = []
    for state in sorted(by_state):
        st = state_stats(by_state[state], rng)
        state_results.append({"state": state, **st})
    eligible = [s for s in state_results if s["eligible"]]
    eligible.sort(key=lambda s: -(s["auc"] if s["auc"] is not None else -1))

    print("\n" + "=" * 78)
    print("2. PER-STATE SENSITIVITY OF THE INDEX (all years; exploratory)")
    print(f"   eligible = >= {STATE_MIN_YEARS} years, >= {STATE_MIN_CLASS} At-Risk and >= {STATE_MIN_CLASS} Low records")
    print("=" * 78)
    print(f"  {'state':<22}{'n':>5}{'pos':>5}{'yrs':>5}{'base':>7}   {'all-years AUC [year-cluster 95% CI]':<38}flag")
    for s in eligible:
        print(f"  {s['state']:<22}{s['n']:>5}{s['n_pos']:>5}{s['n_years']:>5}{s['base_rate']:>7.3f}   "
              f"{fmt_auc(s['auc'], s['ci_lo'], s['ci_hi']):<38}{s['flag']}")
    thin = [s for s in state_results if not s["eligible"]]
    print(f"\n  Not eligible (too little Kharif data): {len(thin)} states -- "
          + (", ".join(s["state"] for s in thin[:8]) + (" ..." if len(thin) > 8 else "") if thin else "none"))

    n_el = len(eligible)
    n_sens = sum(1 for s in eligible if s["flag"] == "SENSITIVE")
    n_rev = sum(1 for s in eligible if s["flag"] == "REVERSED")
    n_above = sum(1 for s in eligible if s["auc"] is not None and s["auc"] > 0.5)
    p_sign = sign_test_one_sided(n_above, n_el)
    exp_fp = FALSE_POSITIVE_RATE * n_el
    print(f"\n  Eligible states: {n_el} | SENSITIVE: {n_sens} | REVERSED: {n_rev}")
    print(f"  States with AUC > 0.5: {n_above} of {n_el}"
          + (f"  (one-sided sign test p = {p_sign:.4f}; states are not fully independent)" if p_sign is not None else ""))
    print(f"  SENSITIVE flags expected by chance alone: about {exp_fp:.1f}  (observed {n_sens})")

    # ---- 3. (state, year) table for a later display layer ----
    state_year = []
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["state"], r["year"])].append(r)
    for (state, year), rs in sorted(grouped.items()):
        state_year.append({"state": state, "year": year, "split": rs[0]["split"],
                           "index_risk_score": round(rs[0]["score"], 4), "tier": rs[0]["tier"],
                           "n_records": len(rs), "at_risk_share": round(sum(r["y"] for r in rs) / len(rs), 4)})

    # ---- 4. Final read ----
    print("\n" + "=" * 78)
    print("3. WHAT THIS MEANS")
    print("=" * 78)
    d = split_reports["test"]
    hi, lo_t = d["tiers"]["High"], d["tiers"]["Low"]
    print(f"  Test: At-Risk rate {lo_t['rate']:.3f} in wet seasons vs {hi['rate']:.3f} in dry seasons "
          f"(overall {d['base_rate']:.3f}).")
    if verdict == "PASS":
        print("  The tiers rank held-out Kharif seasons in the expected order with a CI above zero.")
        print("  Report them as the explainable output: a three-level, state-relative moisture risk rule.")
    elif verdict == "PARTIAL":
        print("  Dry seasons run riskier than wet ones, but the ordering or the CI is not clean on 3 test years.")
        print("  Report the tiers as consistent with the AUC result, not as a calibrated forecast.")
    else:
        print("  The tiers do not separate on the test years. Report that plainly; the all-years AUC still stands.")

    out_path = seq_path.parent / "kharif_suitability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "scope_season": SCOPE_SEASON,
            "index": {"features": MOISTURE_FEATURES, "definition": "-(mean anomaly z); higher = drier = riskier"},
            "reproduces_08_11": all_match,
            "tier_cuts": {"lower": cuts[0], "upper": cuts[1], "fitted_on": "Kharif train rows"},
            "tier_reports": split_reports,
            "tier_verdict_test": verdict,
            "states": state_results,
            "state_summary": {"eligible": n_el, "sensitive": n_sens, "reversed": n_rev,
                              "auc_above_half": n_above, "sign_test_p": p_sign,
                              "expected_false_positives": exp_fp},
            "state_year": state_year,
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()