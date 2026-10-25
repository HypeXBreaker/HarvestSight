"""
11_crop_signal_check.py

HarvestSight -- does focusing on fewer crops (or one crop) help?

WHY THIS EXISTS
---------------
08 found a real but weak At-Risk signal in weather anomalies, strongest in
Kharif moisture (test AUC 0.58-0.59). But 08 pooled every crop within a
season, so it cannot say WHICH crops carry that signal. Rice-Kharif and
Cotton-Kharif do not fail for the same weather reasons; pooling them may be
diluting a clean signal in a few crops -- or the signal may be spread thinly
across all of them, in which case splitting only loses sample size.

This script answers that with NO TRAINING, in a few minutes.

THE SCORE: A PRE-SPECIFIED MOISTURE INDEX (nothing is fitted to labels)
------------------------------------------------------------------------
For each record, risk score = - mean( anomaly z-scores of
    PRECTOTCORR_season_mean, GWETROOT_season_mean, RH2M_season_mean )
i.e. wetter than usual (rain, root-zone wetness, humidity) => LOWER risk.
Anomalies are computed exactly as in 08 (z-score against the (state, season)
climatology fitted on TRAIN years only). Direction and equal weights are
fixed in advance from agronomy and 08's pooled finding; no per-crop feature
or direction picking, so there are no label-fitted parameters to overfit and
no feature-selection multiple-comparison problem within a crop.
This index IS the transparent per-season moisture baseline: a one-line,
fully explainable rule.

HONEST NOTE ON INDEPENDENCE: the choice of these three features was
informed by 08, which looked at all splits. That is a mild
researcher-degrees-of-freedom issue. It is why part 5 below selects crops
using ONLY years <= 2016 and then evaluates once on 2017-19.

WHAT IT PRINTS
--------------
1. Reproduction check: Kharif-pooled test AUC for each of the three features
   must match 08's table (0.594 / 0.586 / 0.581). If it does, the anomaly
   pipeline here is identical to 08's.
2. The index by season (pooled over crops) -- the baseline to beat.
3. Every (season, crop) group with enough data: AUC over ALL years with a
   (state, year) cluster-bootstrap 95% CI, AUC in each time split, and a flag:
      STRONG    AUC >= 0.56, CI lower bound > 0.50, and AUC > 0.5 in at
                least 2 of the evaluable time splits
      LEAD      AUC >= 0.56 and CI lower bound > 0.50, but not consistent
      REVERSED  AUC <= 0.44 with CI upper bound < 0.50 (wetter => MORE risk)
   A group is evaluable only with >= 150 records, >= 40 At-Risk, >= 8 states.
   The expected number of false STRONG/LEAD flags by chance is printed next
   to the observed number.
4. The focus experiment (leakage-free): select crop-season groups using
   ONLY years <= 2016 (same SIGNAL rule), then evaluate the pooled index on
   the held-out 2017-2019 test years, once, against: all groups, the groups
   not selected, and Kharif-pooled.

PRE-REGISTERED READING OF THE FOCUS EXPERIMENT (fixed before running)
---------------------------------------------------------------------
  FOCUS HELPS    selected-groups test AUC >= 0.56, cluster-CI lower bound
                 > 0.50, AND at least 0.02 above the all-groups test AUC
  INCONCLUSIVE   at least 0.02 above all-groups, but CI includes 0.50
  NO HELP        otherwise
  NO GROUP QUALIFIED   nothing passed the selection rule on years <= 2016

CAVEATS
-------
  * Per-crop samples are small (a few hundred records, ~8-15 test
    positives per crop); that is why the headline numbers use ALL years and
    cluster-bootstrap CIs rather than test-only AUCs.
  * Anomaly climatology is fitted on train years, so train-year anomalies are
    slightly in-sample. No label is used anywhere in the index.
  * Many groups are tested; expect a few to look good by chance. The printed
    expected-false-positive count is the yardstick.
  * Still ONE weather point per state.
  * The state-wise suitability idea is deliberately NOT included here; it is
    only worth building if a crop scope with real signal exists.

Input:
  rnn_sequences.json   (the RAW relabeled sequence file that 08 used, NOT the
                        weekly-anomaly file)
  Output:
  crop_signal_results.json  (written next to the input)

Runtime: a few minutes, pure Python, stdlib only.
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
SEASONS = ["Kharif", "Rabi", "Summer"]

N_TEST_YEARS = 3
TRAIN_FRACTION_OF_REMAINDER = 0.85   # identical to 07/08's split
MIN_CLIM_YEARS = 8
BOOTSTRAP_B = 300
RANDOM_SEED = 42
TOP_FRACTION = 0.20

MOISTURE_FEATURES = ["PRECTOTCORR_season_mean", "GWETROOT_season_mean", "RH2M_season_mean"]

MIN_RECORDS = 150
MIN_POSITIVES = 40
MIN_STATES = 8
SIGNAL_AUC = 0.56
REVERSED_AUC = 0.44
SPLIT_MIN_CLASS = 20
FOCUS_GAIN = 0.02
FALSE_POSITIVE_RATE = 0.025   # one-sided 2.5% from the CI-lower-bound > 0.5 requirement

EXPECTED_RECORDS = 10261
EXPECTED_CELLS = 1377
# 08's Kharif-pooled TEST AUCs for these features (direction: wetter => less risk)
BENCH_08_KHARIF_TEST = {"PRECTOTCORR_season_mean": 0.594, "GWETROOT_season_mean": 0.586,
                        "RH2M_season_mean": 0.581}


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
# Split, features, climatology (identical to 08)
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


# ---------------------------------------------------------------------------
# Metrics (identical to 08 / 07 v3)
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


def cluster_bootstrap_ci(items, rng, b=BOOTSTRAP_B):
    """95% CI for AUC, resampling whole (state, year) clusters.
    items: list of (cluster_id, label, score). Returns (lo, hi) or (None, None)."""
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
    aucs.sort()
    last = len(aucs) - 1
    return aucs[int(0.025 * last)], aucs[int(0.975 * last)]


def top_fraction_precision(true_labels, scores, positive_idx, fraction=TOP_FRACTION):
    """Precision among the top `fraction` of records by score (ties broken by
    original order, so the result is deterministic). Threshold-free."""
    n = len(true_labels)
    k = max(1, int(math.ceil(fraction * n)))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    hits = sum(1 for i in order[:k] if true_labels[i] == positive_idx)
    n_pos = sum(1 for t in true_labels if t == positive_idx)
    base_rate = n_pos / n
    precision = hits / k
    return {
        "k": k, "hits": hits, "precision": precision,
        "recall": (hits / n_pos) if n_pos else 0.0,
        "base_rate": base_rate,
        "lift": (precision / base_rate) if base_rate else 0.0,
    }


# ---------------------------------------------------------------------------
# New in 11: moisture index, group statistics, focus experiment
# ---------------------------------------------------------------------------

def moisture_index(anomalies):
    """Mean of the available moisture anomalies (higher = wetter than usual).
    None if none of the three is available."""
    vals = [anomalies[f] for f in MOISTURE_FEATURES if anomalies.get(f) is not None]
    return (sum(vals) / len(vals)) if vals else None


def group_stats(rows, rng, with_split_detail=True):
    """Statistics for one group of rows. Each row needs: y, score, cluster,
    state, split. score is the RISK score (higher = more At-Risk)."""
    n = len(rows)
    ys = [r["y"] for r in rows]
    n_pos = sum(ys)
    n_states = len({r["state"] for r in rows})
    out = {"n": n, "n_pos": n_pos, "n_states": n_states,
           "base_rate": (n_pos / n) if n else None}
    out["eligible"] = n >= MIN_RECORDS and n_pos >= MIN_POSITIVES and n_states >= MIN_STATES
    out["auc"] = roc_auc(ys, [r["score"] for r in rows], 1) if n else None
    out["ci_lo"] = out["ci_hi"] = None
    if out["eligible"] and out["auc"] is not None:
        lo, hi = cluster_bootstrap_ci([(r["cluster"], r["y"], r["score"]) for r in rows], rng)
        out["ci_lo"], out["ci_hi"] = lo, hi
    auc, lo, hi = out["auc"], out["ci_lo"], out["ci_hi"]
    out["signal"] = bool(out["eligible"] and auc is not None and lo is not None
                         and auc >= SIGNAL_AUC and lo > 0.5)
    out["reversed"] = bool(out["eligible"] and auc is not None and hi is not None
                           and auc <= REVERSED_AUC and hi < 0.5)
    if with_split_detail:
        split_aucs = {}
        for sp in ("train", "val", "test"):
            sub = [r for r in rows if r["split"] == sp]
            pos = sum(r["y"] for r in sub)
            if pos >= SPLIT_MIN_CLASS and (len(sub) - pos) >= SPLIT_MIN_CLASS:
                split_aucs[sp] = roc_auc([r["y"] for r in sub], [r["score"] for r in sub], 1)
            else:
                split_aucs[sp] = None
        evaluable = [a for a in split_aucs.values() if a is not None]
        out["split_aucs"] = split_aucs
        out["consistent"] = len(evaluable) >= 2 and sum(1 for a in evaluable if a > 0.5) >= 2
    return out


def flag_for(stats):
    if stats.get("signal") and stats.get("consistent"):
        return "STRONG"
    if stats.get("signal"):
        return "LEAD"
    if stats.get("reversed"):
        return "REVERSED"
    return "-"


def pooled_eval(rows, rng):
    """AUC (+ cluster CI) and top-20% precision/lift for a pooled set of rows."""
    n = len(rows)
    out = {"n": n, "n_pos": sum(r["y"] for r in rows), "base_rate": None, "auc": None,
           "ci_lo": None, "ci_hi": None, "top_precision": None, "lift": None}
    if n == 0:
        return out
    ys = [r["y"] for r in rows]
    sc = [r["score"] for r in rows]
    out["base_rate"] = out["n_pos"] / n
    out["auc"] = roc_auc(ys, sc, 1)
    if out["auc"] is not None:
        out["ci_lo"], out["ci_hi"] = cluster_bootstrap_ci(
            [(r["cluster"], r["y"], r["score"]) for r in rows], rng)
        top = top_fraction_precision(ys, sc, 1)
        out["top_precision"], out["lift"] = top["precision"], top["lift"]
    return out


def focus_experiment(groups, rng):
    """Select groups using ONLY train+val rows (years <= 2016), then evaluate the
    pooled index on the held-out test years, once. `groups`: (season, crop) -> rows."""
    selected = []
    selection_table = []
    for key in sorted(groups):
        tv = [r for r in groups[key] if r["split"] in ("train", "val")]
        st = group_stats(tv, rng, with_split_detail=False)
        selection_table.append({"season": key[0], "crop": key[1], "n": st["n"], "n_pos": st["n_pos"],
                                "eligible": st["eligible"], "auc": st["auc"], "ci_lo": st["ci_lo"],
                                "ci_hi": st["ci_hi"], "selected": st["signal"]})
        if st["signal"]:
            selected.append(key)
    sel_set = set(selected)

    def test_rows(pred):
        return [r for k, rows in groups.items() if pred(k) for r in rows if r["split"] == "test"]

    comparisons = {
        "selected": pooled_eval(test_rows(lambda k: k in sel_set), rng),
        "all_groups": pooled_eval(test_rows(lambda k: True), rng),
        "rest_not_selected": pooled_eval(test_rows(lambda k: k not in sel_set), rng),
        "kharif_pooled": pooled_eval(test_rows(lambda k: k[0] == "Kharif"), rng),
    }
    sel, allg = comparisons["selected"], comparisons["all_groups"]
    if not selected:
        verdict = "NO GROUP QUALIFIED"
    elif (sel["auc"] is not None and allg["auc"] is not None
          and sel["auc"] >= SIGNAL_AUC and sel["ci_lo"] is not None and sel["ci_lo"] > 0.5
          and sel["auc"] >= allg["auc"] + FOCUS_GAIN):
        verdict = "FOCUS HELPS"
    elif (sel["auc"] is not None and allg["auc"] is not None
          and sel["auc"] >= allg["auc"] + FOCUS_GAIN):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO HELP"
    return {"selected_groups": selected, "selection_table": selection_table,
            "comparisons": comparisons, "verdict": verdict}


def fmt_auc(auc, lo=None, hi=None):
    if auc is None:
        return "   --  "
    s = f"{auc:.3f}"
    if lo is not None and hi is not None:
        s += f" [{lo:.3f},{hi:.3f}]"
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
        raise ValueError("This is the weekly-ANOMALY file. 11 (like 08) needs the RAW relabeled "
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
    skipped = 0
    for r in records:
        an = anomalies.get((r["state"], r["season"], r["year"]))
        idx = moisture_index(an) if an is not None else None
        if idx is None:
            skipped += 1
            continue
        rows_all.append({"state": r["state"], "crop": r["crop"], "season": r["season"], "year": r["year"],
                         "split": split_of_year[r["year"]], "cluster": (r["state"], r["year"]),
                         "y": 1 if r["risk"] == "At-Risk" else 0, "score": -idx, "an": an})
    print(f"Rows scored: {len(rows_all):,}  (skipped {skipped:,} without a usable moisture index)")

    # ---- 1. Reproduction check against 08 ----
    print("\n" + "=" * 78)
    print("1. REPRODUCTION CHECK vs 08 (Kharif-pooled, TEST split, wetter => less risk)")
    print("=" * 78)
    kh_test = [r for r in rows_all if r["season"] == "Kharif" and r["split"] == "test"]
    all_match = True
    for f in MOISTURE_FEATURES:
        sub = [(r["y"], -r["an"][f]) for r in kh_test if r["an"].get(f) is not None]
        a = roc_auc([y for y, _ in sub], [s for _, s in sub], 1)
        ok = a is not None and abs(a - BENCH_08_KHARIF_TEST[f]) < 0.002
        all_match = all_match and ok
        print(f"  {f:<26} AUC {a:.3f}   08 reported {BENCH_08_KHARIF_TEST[f]:.3f}   "
              f"{'MATCH' if ok else 'DIFFERS'}")
    print("  " + ("Anomaly pipeline reproduces 08 exactly." if all_match else
                  "WARNING: does not reproduce 08 -- check the input file before trusting anything below."))

    # ---- 2. Index by season ----
    print("\n" + "=" * 78)
    print("2. MOISTURE INDEX BY SEASON (pooled over crops) -- the transparent baseline")
    print("=" * 78)
    print(f"  {'group':<14}{'n':>6}{'At-Risk':>8}   {'all-years AUC [cluster 95% CI]':<34}"
          f"{'train':>7}{'val':>7}{'test':>7}")
    season_stats = {}
    for label, pred in [("ALL seasons", lambda r: True)] + [(s, (lambda s: lambda r: r["season"] == s)(s)) for s in SEASONS]:
        sub = [r for r in rows_all if pred(r)]
        st = group_stats(sub, rng)
        season_stats[label] = st
        sa = st["split_aucs"]
        print(f"  {label:<14}{st['n']:>6,}{st['n_pos']:>8,}   {fmt_auc(st['auc'], st['ci_lo'], st['ci_hi']):<34}"
              f"{fmt_auc(sa['train']):>7}{fmt_auc(sa['val']):>7}{fmt_auc(sa['test']):>7}")

    # ---- 3. Per (season, crop) groups ----
    groups = defaultdict(list)
    for r in rows_all:
        groups[(r["season"], r["crop"])].append(r)

    print("\n" + "=" * 78)
    print("3. EVERY (SEASON, CROP) GROUP WITH ENOUGH DATA  (all years; ranked by AUC)")
    print(f"   evaluable = >= {MIN_RECORDS} records, >= {MIN_POSITIVES} At-Risk, >= {MIN_STATES} states")
    print("=" * 78)
    group_results = []
    for key in sorted(groups):
        st = group_stats(groups[key], rng)
        group_results.append({"season": key[0], "crop": key[1], "flag": flag_for(st), **st})
    evaluable = [g for g in group_results if g["eligible"]]
    evaluable.sort(key=lambda g: -(g["auc"] if g["auc"] is not None else -1))
    print(f"  {'group':<28}{'n':>5}{'pos':>5}{'st':>4}   {'all-years AUC [cluster 95% CI]':<30}"
          f"{'train':>7}{'val':>7}{'test':>7}  flag")
    for g in evaluable:
        sa = g["split_aucs"]
        label = f"{g['season']}/{g['crop']}"
        print(f"  {label:<28}{g['n']:>5}{g['n_pos']:>5}{g['n_states']:>4}   "
              f"{fmt_auc(g['auc'], g['ci_lo'], g['ci_hi']):<30}"
              f"{fmt_auc(sa['train']):>7}{fmt_auc(sa['val']):>7}{fmt_auc(sa['test']):>7}  {g['flag']}")
    thin = [g for g in group_results if not g["eligible"]]
    print(f"\n  Not evaluable (too little data): {len(thin)} groups, "
          f"{sum(g['n'] for g in thin):,} records -- e.g. "
          + ", ".join(f"{g['season']}/{g['crop']}" for g in thin[:6]) + (" ..." if len(thin) > 6 else ""))

    n_strong = sum(1 for g in evaluable if g["flag"] == "STRONG")
    n_lead = sum(1 for g in evaluable if g["flag"] == "LEAD")
    n_rev = sum(1 for g in evaluable if g["flag"] == "REVERSED")
    exp_fp = FALSE_POSITIVE_RATE * len(evaluable)
    print(f"\n  Evaluable groups: {len(evaluable)} | STRONG: {n_strong} | LEAD: {n_lead} | REVERSED: {n_rev}")
    print(f"  STRONG+LEAD expected by pure chance: about {exp_fp:.1f}  (observed {n_strong + n_lead})")

    # ---- 4. Focus experiment ----
    print("\n" + "=" * 78)
    print("4. FOCUS EXPERIMENT -- select groups on years <= 2016 only, evaluate ONCE on 2017-19")
    print("=" * 78)
    fx = focus_experiment(groups, rng)
    sel = fx["selected_groups"]
    print(f"  Groups selected on train+val (AUC >= {SIGNAL_AUC}, cluster-CI lower bound > 0.50): {len(sel)}")
    for key in sel:
        row = next(t for t in fx["selection_table"] if (t["season"], t["crop"]) == key)
        print(f"    {key[0]}/{key[1]:<26} train+val AUC {fmt_auc(row['auc'], row['ci_lo'], row['ci_hi'])}")
    print(f"\n  {'held-out TEST set':<24}{'n':>6}{'At-Risk':>8}{'base':>7}   {'AUC [cluster 95% CI]':<26}{'top-20% lift':>13}")
    for name in ("selected", "all_groups", "rest_not_selected", "kharif_pooled"):
        c = fx["comparisons"][name]
        base = f"{c['base_rate']:.3f}" if c["base_rate"] is not None else "  --"
        lift = f"x{c['lift']:.2f}" if c["lift"] is not None else "  --"
        print(f"  {name:<24}{c['n']:>6,}{c['n_pos']:>8,}{base:>7}   "
              f"{fmt_auc(c['auc'], c['ci_lo'], c['ci_hi']):<26}{lift:>13}")
    print(f"\n  FOCUS VERDICT (pre-registered): {fx['verdict']}")

    # ---- 5. Final read ----
    print("\n" + "=" * 78)
    print("5. WHAT THIS MEANS")
    print("=" * 78)
    k_all = season_stats["Kharif"]
    print(f"  Transparent baseline, Kharif-pooled moisture index: all-years AUC {fmt_auc(k_all['auc'], k_all['ci_lo'], k_all['ci_hi'])}")
    print("  (compare: the RNN's best held-out result was test AUC 0.529 [0.475, 0.593]).")
    if fx["verdict"] == "FOCUS HELPS":
        print("  Focusing on the selected crop-season groups improves held-out ranking. Build the final")
        print("  deliverable on THAT scope (the moisture index, optionally an RNN restricted to those")
        print("  groups), and report the selection procedure alongside the test result.")
    elif fx["verdict"] == "INCONCLUSIVE":
        print("  The selected groups look better on test but the interval includes chance. Report as a")
        print("  lead, not a result; the Kharif-pooled moisture index remains the safer headline.")
    elif fx["verdict"] == "NO GROUP QUALIFIED":
        print("  No single crop-season showed a clear moisture signal on train+val. Per-crop splitting")
        print("  loses more sample size than it gains in specificity; use the season-level index.")
    else:
        print("  Focusing on fewer crops does not improve held-out ranking over using all groups. The")
        print("  Kharif-pooled moisture index is the honest headline; report the RNN as tested and ruled out.")

    out_path = seq_path.parent / "crop_signal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "moisture_features": MOISTURE_FEATURES,
            "reproduces_08": all_match,
            "season_baseline": {k: {kk: vv for kk, vv in v.items()} for k, v in season_stats.items()},
            "groups": group_results,
            "n_evaluable": len(evaluable), "n_strong": n_strong, "n_lead": n_lead, "n_reversed": n_rev,
            "expected_false_positives": exp_fp,
            "focus_experiment": fx,
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()