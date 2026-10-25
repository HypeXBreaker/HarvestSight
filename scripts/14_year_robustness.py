"""
14_year_robustness.py

HarvestSight -- do the Kharif risk tiers hold up when the unit is the YEAR?

WHY THIS EXISTS
---------------
13's tier verdict was PASS on the 3 test years, but its CI resampled
(state, year) clusters. Monsoon anomalies are national, so states in the same
year are not independent; the effective test sample is closer to 3 years than
to 75 clusters. This script removes that weakness by making the YEAR the unit
of evidence, using all 23 years. NO TRAINING. Nothing is fitted to labels.

SETUP (identical to 13, verified by reproduction checks)
--------------------------------------------------------
Same index (-(mean anomaly z of rainfall, root-zone wetness, humidity)),
climatology fitted on train years, tier cut points = train-year terciles of
Kharif rows. Cut points are FIXED; they are never refitted per year.

WHAT IT COMPUTES
----------------
1. Per-year tier contrast: for every year, the At-Risk rate among High-tier
   (dry) rows vs Low-tier (wet) rows. A year is usable if both tiers have
   >= 15 rows.
2. SIGN TEST across usable years: in how many years is the High-tier rate
   above the Low-tier rate? One-sided exact binomial p (H0: 50/50). Ties are
   excluded. Also shown separately for train / val / test years (descriptive).
3. YEAR-CLUSTER BOOTSTRAP (resample whole years) 95% CIs for
     (a) the pooled High-minus-Low At-Risk rate difference (primary), and
     (b) the mean of the per-year differences (equal weight per year).
4. NATIONAL YEAR-LEVEL RANK CORRELATION (secondary, descriptive): across the
   23 years, Spearman rho between the national mean index (mean over
   state-years) and the national At-Risk share, with a one-sided permutation
   p-value. This is the cleanest "drier national Kharif => more At-Risk" test
   because it has exactly one point per year.

PRE-REGISTERED VERDICT (fixed before running)
---------------------------------------------
  ROBUST   sign-test p < 0.05 AND year-cluster CI lower bound of the pooled
           High-minus-Low difference > 0
  PARTIAL  exactly one of those two holds
  FAIL     neither holds
The rank correlation is reported but NOT part of the verdict.

CAVEATS
-------
  * 20 of the 23 years are train/val years, and 08 chose the three features
    after looking at all splits. This check shows the tiers are not an
    artifact of 3 lucky test years; it is not a clean out-of-sample test of
    the feature choice.
  * Years are not independent either (multi-year dry spells, trends); the
    sign test treats them as independent.
  * Cut points and climatology come from train years, so train-year tiers are
    in-sample for those two (label-free) quantities only.
  * With only 3 test years, nothing about the test years alone can be
    inferred from this script; they are listed for completeness.
  * Still ONE weather point per state; state-aggregated yields.

Input:
  rnn_sequences.json   (the RAW relabeled sequence file, not the anomaly one)
Output:
  kharif_year_robustness.json  (written next to the input)

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
TRAIN_FRACTION_OF_REMAINDER = 0.85   # identical to 07/08/11/13
MIN_CLIM_YEARS = 8
RANDOM_SEED = 42

MOISTURE_FEATURES = ["PRECTOTCORR_season_mean", "GWETROOT_season_mean", "RH2M_season_mean"]

MIN_TIER_ROWS = 15            # per tier, per year, for a year to be usable
SIGN_ALPHA = 0.05
BOOTSTRAP_B = 5000
PERMUTATIONS = 20000

EXPECTED_RECORDS = 10261
EXPECTED_CELLS = 1377
EXPECTED_KHARIF_RECORDS = 5445
BENCH_08_KHARIF_TEST = {"PRECTOTCORR_season_mean": 0.594, "GWETROOT_season_mean": 0.586,
                        "RH2M_season_mean": 0.581}
BENCH_11_KHARIF_ALLYEARS_AUC = 0.608
BENCH_13_CUTS = (-0.410, 0.418)
BENCH_13_TEST_RATES = {"Low": 0.177, "Medium": 0.289, "High": 0.349}


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
# Split, features, climatology (identical to 08 / 11 / 13)
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


# ---------------------------------------------------------------------------
# New in 14: year-level statistics
# ---------------------------------------------------------------------------

def _ci95(vals):
    vals = sorted(vals)
    last = len(vals) - 1
    return vals[int(0.025 * last)], vals[int(0.975 * last)]


def year_table(rows, split_of_year):
    """One entry per year: tier counts, At-Risk rates, High-minus-Low difference
    (None if the year is not usable), national mean index and At-Risk share."""
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    table = []
    for y in sorted(by_year):
        rs = by_year[y]
        low = [r for r in rs if r["tier"] == "Low"]
        high = [r for r in rs if r["tier"] == "High"]
        n_low, n_high = len(low), len(high)
        pos_low, pos_high = sum(r["y"] for r in low), sum(r["y"] for r in high)
        usable = n_low >= MIN_TIER_ROWS and n_high >= MIN_TIER_ROWS
        rate_low = (pos_low / n_low) if n_low else None
        rate_high = (pos_high / n_high) if n_high else None
        state_score = {}
        for r in rs:
            state_score[r["state"]] = r["score"]    # one index value per state-year
        table.append({
            "year": y, "split": split_of_year[y], "n": len(rs),
            "n_low": n_low, "pos_low": pos_low, "rate_low": rate_low,
            "n_high": n_high, "pos_high": pos_high, "rate_high": rate_high,
            "usable": usable,
            "diff": (rate_high - rate_low) if usable else None,
            "national_index": sum(state_score.values()) / len(state_score),
            "national_at_risk": sum(r["y"] for r in rs) / len(rs),
        })
    return table


def sign_test_one_sided(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return None
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def sign_summary(entries):
    """Counts over usable years: High>Low, High<Low, ties; one-sided sign-test p."""
    diffs = [e["diff"] for e in entries if e["diff"] is not None]
    up = sum(1 for d in diffs if d > 0)
    down = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - up - down
    p = sign_test_one_sided(up, up + down) if (up + down) > 0 else None
    return {"usable_years": len(diffs), "high_above_low": up, "high_below_low": down,
            "ties": ties, "sign_p_one_sided": p}


def year_bootstrap(table, rng, b=None):
    """Resample whole YEARS. Returns 95% CIs for (a) the pooled High-minus-Low
    rate difference and (b) the mean of the per-year differences."""
    b = BOOTSTRAP_B if b is None else b
    n = len(table)
    pooled, means = [], []
    for _ in range(b):
        sample = [table[rng.randrange(n)] for _ in range(n)]
        n_low = sum(t["n_low"] for t in sample)
        n_high = sum(t["n_high"] for t in sample)
        if n_low and n_high:
            pooled.append(sum(t["pos_high"] for t in sample) / n_high
                          - sum(t["pos_low"] for t in sample) / n_low)
        diffs = [t["diff"] for t in sample if t["diff"] is not None]
        if len(diffs) >= 5:
            means.append(sum(diffs) / len(diffs))
    return {"pooled": _ci95(pooled) if len(pooled) >= 100 else None,
            "mean_of_years": _ci95(means) if len(means) >= 100 else None}


def pooled_diff(entries):
    n_low = sum(t["n_low"] for t in entries)
    n_high = sum(t["n_high"] for t in entries)
    if not n_low or not n_high:
        return None
    return sum(t["pos_high"] for t in entries) / n_high - sum(t["pos_low"] for t in entries) / n_low


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman_perm(x, y, rng, n_perm=None):
    """Spearman rho and a ONE-SIDED (rho > 0) permutation p-value."""
    n_perm = PERMUTATIONS if n_perm is None else n_perm
    rx, ry = average_ranks(x), average_ranks(y)
    rho = pearson(rx, ry)
    perm = ry[:]
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        if pearson(rx, perm) >= rho - 1e-12:
            hits += 1
    return rho, (hits + 1) / (n_perm + 1)


def verdict_for(sign_p, boot_pooled):
    """Pre-registered reading (see module docstring)."""
    sign_ok = sign_p is not None and sign_p < SIGN_ALPHA
    ci_ok = boot_pooled is not None and boot_pooled[0] > 0
    if sign_ok and ci_ok:
        return "ROBUST"
    if sign_ok or ci_ok:
        return "PARTIAL"
    return "FAIL"


def fmt_rate(x):
    return f"{x:.3f}" if x is not None else "  -- "


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
        raise ValueError("This is the weekly-ANOMALY file. 14 (like 08/11/13) needs the RAW relabeled "
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
        rows_all.append({"state": r["state"], "season": r["season"], "year": r["year"],
                         "split": split_of_year[r["year"]],
                         "y": 1 if r["risk"] == "At-Risk" else 0, "score": -idx, "an": an})
    rows = [r for r in rows_all if r["season"] == SCOPE_SEASON]
    print(f"Rows scored: {len(rows_all):,} of {len(records):,}  |  {SCOPE_SEASON} rows: {len(rows):,} "
          f"(benchmark {EXPECTED_KHARIF_RECORDS:,})")

    # ---- 0. Reproduction checks (08 / 11 / 13) ----
    print("\n" + "=" * 78)
    print("0. REPRODUCTION CHECKS (Kharif)")
    print("=" * 78)
    all_match = True
    kh_test = [r for r in rows if r["split"] == "test"]
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

    cuts = tercile_cuts([r["score"] for r in rows if r["split"] == "train"])
    for r in rows:
        r["tier"] = tier_of(r["score"], cuts)
    ok = abs(cuts[0] - BENCH_13_CUTS[0]) < 0.002 and abs(cuts[1] - BENCH_13_CUTS[1]) < 0.002
    all_match = all_match and ok
    print(f"  tier cut points   {cuts[0]:+.3f} / {cuts[1]:+.3f}   13 reported "
          f"{BENCH_13_CUTS[0]:+.3f} / {BENCH_13_CUTS[1]:+.3f}   {'MATCH' if ok else 'DIFFERS'}")
    for t in ("Low", "Medium", "High"):
        sub = [r for r in kh_test if r["tier"] == t]
        rate = sum(r["y"] for r in sub) / len(sub) if sub else None
        ok = rate is not None and abs(rate - BENCH_13_TEST_RATES[t]) < 0.002
        all_match = all_match and ok
        print(f"  test At-Risk rate, {t:<7} tier {fmt_rate(rate)}   13 reported {BENCH_13_TEST_RATES[t]:.3f}   "
              f"{'MATCH' if ok else 'DIFFERS'}")
    print("  " + ("Pipeline reproduces 08, 11 and 13 exactly." if all_match else
                  "WARNING: does not reproduce earlier scripts -- check the input file before trusting anything below."))

    # ---- 1. Per-year contrast ----
    table = year_table(rows, split_of_year)
    print("\n" + "=" * 78)
    print(f"1. PER-YEAR TIER CONTRAST (cut points fixed from train; usable = >= {MIN_TIER_ROWS} rows per tier)")
    print("=" * 78)
    print(f"  {'year':<6}{'split':<7}{'n':>5}{'nLow':>6}{'nHigh':>7}{'Low rate':>10}{'High rate':>11}"
          f"{'High-Low':>10}{'nat. idx':>10}{'nat. AtRisk':>13}")
    for t in table:
        diff = f"{t['diff']:+.3f}" if t["diff"] is not None else "  n/u"
        print(f"  {t['year']:<6}{t['split']:<7}{t['n']:>5}{t['n_low']:>6}{t['n_high']:>7}"
              f"{fmt_rate(t['rate_low']):>10}{fmt_rate(t['rate_high']):>11}{diff:>10}"
              f"{t['national_index']:>+10.3f}{t['national_at_risk']:>13.3f}")
    n_unusable = sum(1 for t in table if not t["usable"])
    print(f"\n  n/u = not usable (a tier has < {MIN_TIER_ROWS} rows that year). "
          f"Unusable years: {n_unusable} of {len(table)}.")
    print("  nat. idx = mean index over states (higher = drier); nat. AtRisk = share of Kharif rows At-Risk.")

    # ---- 2. Sign test ----
    print("\n" + "=" * 78)
    print("2. SIGN TEST: in how many years is the High-tier (dry) At-Risk rate above the Low-tier (wet) rate?")
    print("=" * 78)
    sign_all = sign_summary(table)
    by_split = {sp: sign_summary([t for t in table if t["split"] == sp]) for sp in ("train", "val", "test")}
    print(f"  {'years':<10}{'usable':>8}{'High>Low':>10}{'High<Low':>10}{'ties':>6}{'one-sided p':>14}")
    for name, s in (("ALL", sign_all), ("train", by_split["train"]), ("val", by_split["val"]),
                    ("test", by_split["test"])):
        p = f"{s['sign_p_one_sided']:.4f}" if s["sign_p_one_sided"] is not None else "   --"
        print(f"  {name:<10}{s['usable_years']:>8}{s['high_above_low']:>10}{s['high_below_low']:>10}"
              f"{s['ties']:>6}{p:>14}")
    print("  (train/val/test rows are descriptive; test has only 3 years, so no inference from it alone.)")

    # ---- 3. Year-cluster bootstrap ----
    boot = year_bootstrap(table, rng)
    pooled_all = pooled_diff(table)
    usable_diffs = [t["diff"] for t in table if t["diff"] is not None]
    mean_year_diff = (sum(usable_diffs) / len(usable_diffs)) if usable_diffs else None
    print("\n" + "=" * 78)
    print(f"3. YEAR-CLUSTER BOOTSTRAP ({BOOTSTRAP_B:,} resamples of whole years)")
    print("=" * 78)

    def ci_txt(point, ci):
        if point is None or ci is None:
            return "--"
        return f"{point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"

    print(f"  pooled High-minus-Low At-Risk rate (primary)   {ci_txt(pooled_all, boot['pooled'])}")
    print(f"  mean of per-year differences (equal weight)    {ci_txt(mean_year_diff, boot['mean_of_years'])}")
    print(f"  (13's (state, year)-cluster CI on test was +0.172 [+0.051, +0.297]; "
          f"compare the width of the year-level CI above.)")

    # ---- 4. National year-level rank correlation ----
    xs = [t["national_index"] for t in table]
    ys_nat = [t["national_at_risk"] for t in table]
    rho, p_perm = spearman_perm(xs, ys_nat, rng)
    print("\n" + "=" * 78)
    print(f"4. NATIONAL YEAR-LEVEL RANK CORRELATION (n = {len(table)} years; descriptive, not in the verdict)")
    print("=" * 78)
    print(f"  Spearman rho between national mean index (drier = higher) and national At-Risk share: "
          f"{rho:+.3f}   one-sided permutation p = {p_perm:.4f}")

    # ---- 5. Verdict ----
    verdict = verdict_for(sign_all["sign_p_one_sided"], boot["pooled"])
    print("\n" + "=" * 78)
    print("5. VERDICT (pre-registered)")
    print("=" * 78)
    print(f"  ROBUST   sign-test p < {SIGN_ALPHA} AND year-cluster CI lower bound of pooled High-minus-Low > 0")
    print("  PARTIAL  exactly one of the two")
    print("  FAIL     neither")
    sp = sign_all["sign_p_one_sided"]
    print(f"  sign test: {sign_all['high_above_low']} of {sign_all['high_above_low'] + sign_all['high_below_low']} "
          f"usable years High>Low, p = {sp:.4f}" if sp is not None else "  sign test: not computable")
    print(f"  year-cluster CI lower bound: {boot['pooled'][0]:+.3f}" if boot["pooled"] else "  CI: not computable")
    print(f"  --> {verdict}")
    if verdict == "ROBUST":
        print("  The dry-vs-wet separation holds across years, not just on the 3 test years. The tier rule is")
        print("  supported as a historical pattern; state it as such (feature choice came from 08, which saw all splits).")
    elif verdict == "PARTIAL":
        print("  Evidence points the right way but one criterion fails. Report the tiers as consistent with, not proof of,")
        print("  a year-level relationship, and quote both numbers.")
    else:
        print("  The year-level evidence does not support the tiers. The 13 PASS rested on 3 correlated test years; report that.")

    out_path = seq_path.parent / "kharif_year_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "scope_season": SCOPE_SEASON,
            "reproduces_earlier_scripts": all_match,
            "tier_cuts": {"lower": cuts[0], "upper": cuts[1]},
            "min_tier_rows": MIN_TIER_ROWS,
            "years": table,
            "sign_test": {"all": sign_all, **by_split},
            "bootstrap": {"pooled_high_minus_low": {"point": pooled_all, "ci95": boot["pooled"]},
                          "mean_of_year_differences": {"point": mean_year_diff, "ci95": boot["mean_of_years"]}},
            "national_rank_correlation": {"spearman_rho": rho, "perm_p_one_sided": p_perm,
                                          "n_years": len(table)},
            "verdict": verdict,
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()