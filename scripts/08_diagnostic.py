"""
08_signal_diagnostic.py

HarvestSight -- Phase 6 diagnostic: is there ANY usable At-Risk signal in the
weather, once the seasonal cycle is removed?

WHY THIS EXISTS
---------------
After the labels were detrended (05 v2) and the RNN retrained (07 v2), the
model still showed no At-Risk skill on the test years (ROC-AUC 0.489, 95% CI
[0.454, 0.525]; precision 0.205 vs a 0.230 base rate). Before touching the
RNN again we need to know WHERE the information is missing:

  * If weather anomalies DO predict At-Risk out of sample, the signal exists
    and the RNN's raw per-state-normalized input is just burying it under the
    seasonal cycle -> rebuild the RNN input as anomalies.
  * If they DON'T, the state-level (often capital-city) weather point carries
    no usable signal for state-wide yield -> the fix is data (spatial
    resolution, more variables), and more RNN tuning on this input is wasted.

WHAT IT DOES (no RNN, nothing trained that ships)
-------------------------------------------------
1. Reads rnn_sequences.json and applies the SAME temporal split as 07
   (train 1997-2014 / val 2015-16 / test 2017-19).
2. Reduces each record's weekly weather to 25 season-level features, using
   FULL 7-day weeks only (the final partial week is dropped so season length
   can't leak in):
       for each of the 6 weather params: mean over the whole season and over
       the early / middle / late thirds (6 x 4 = 24 features)
       + peak weekly rainfall (1 feature)
3. Converts each feature to an ANOMALY: a z-score against that
   (state, season)'s own climatology, fitted on TRAIN-year cells only. This
   removes the monsoon/winter cycle and each state's normal level, leaving
   "wetter/hotter/drier than usual for this state and season".
4. Measures each anomaly's univariate ROC-AUC against the At-Risk label.
   The direction (higher anomaly -> more or less risk) is chosen on TRAIN
   only and then applied unchanged to val and test, so out-of-sample numbers
   are honest. Reported pooled and per season (Kharif / Rabi / Summer), since
   the mechanisms differ (e.g. rainfall deficit in Kharif, terminal heat in
   Rabi).
5. Fits a small transparent logistic regression on all 25 anomalies PER
   SEASON (diagnostic only, not a candidate model) to test whether a
   combination carries signal that no single feature does.
   It is crop-agnostic by design: it asks whether weather anomalies alone
   carry signal, not whether crop-specific responses add more.

PRE-REGISTERED DECISION RULE (fixed before running)
---------------------------------------------------
A feature (or a per-season combined model) PASSES if, on the held-out data:
    test AUC >= 0.55  AND  val AUC >= 0.53  AND  the 95% cluster-bootstrap
    CI lower bound of the test AUC > 0.50.
VERDICT:
    SIGNAL PRESENT   if any per-season combined model passes OR >= 3
                     individual features pass
    AMBIGUOUS        if 1-2 individual features pass and no combined model
                     does (about 100 individual tests are run, so a lone
                     pass is within what chance produces)
    NO USABLE SIGNAL otherwise

HOW TO READ THE UNCERTAINTY
---------------------------
Every crop in a (state, season, year) shares the same weather, so the
effective sample is the number of (state, season, year) cells, not the
number of records. Confidence intervals therefore resample (state, year)
CLUSTERS, not individual records, which is wider and more honest than the
independent-record interval used in 07. Roughly 100 tests are run in
total; expect a few to look good by chance, which is why the decision
rule asks for several passes or a passing combined model.

KNOWN LIMITS (stated, not hidden)
---------------------------------
  * Anomalies are measured against the TRAIN-period mean; no climate trend
    is removed, so late years can read slightly warm. Labels are detrended,
    so this adds noise rather than a spurious signal.
  * Season-level means can wash out short extreme events (a one-week heat
    spike); the peak weekly rainfall feature is the only extreme-value
    feature included.
  * The weather is still ONE point per state, so this tests the signal in
    that point, not weather in general.

Input:
  rnn_sequences.json   (Phase 5 output; the 10,261-record relabeled file)

Output:
  signal_diagnostic_results.json  (written next to rnn_sequences.json)

Runtime: pure Python, roughly 2-4 minutes. Stdlib only.
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
TRAIN_FRACTION_OF_REMAINDER = 0.85   # identical to 07's split

MIN_CLIM_YEARS = 8        # train-year cells needed to fit a (state, season) climatology
MIN_CLASS_COUNT = 20      # positives AND negatives required per split to report an AUC
BOOTSTRAP_B = 300         # cluster-bootstrap resamples per confidence interval
RANDOM_SEED = 42

PASS_TEST_AUC = 0.55
PASS_VAL_AUC = 0.53

LOGIT_L2 = 0.05
LOGIT_LR = 0.5
LOGIT_ITERS = 200


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
# Split (identical to 07)
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
# Season-level features and anomalies
# ---------------------------------------------------------------------------

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
# Metrics
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


def passes_rule(val_auc, test_auc, ci_lo):
    return (test_auc is not None and val_auc is not None and ci_lo is not None
            and test_auc >= PASS_TEST_AUC and val_auc >= PASS_VAL_AUC and ci_lo > 0.5)


# ---------------------------------------------------------------------------
# Transparent logistic regression (diagnostic baseline only)
# ---------------------------------------------------------------------------

def fit_logistic(X, y, l2=LOGIT_L2, lr=LOGIT_LR, iters=LOGIT_ITERS):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    bias = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = bias + sum(wj * xj for wj, xj in zip(w, xi))
            z = 30.0 if z > 30.0 else (-30.0 if z < -30.0 else z)
            err = 1.0 / (1.0 + math.exp(-z)) - yi
            gb += err
            for j in range(d):
                gw[j] += err * xi[j]
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        bias -= lr * gb / n
    return w, bias


def logit_scores(X, w, bias):
    return [bias + sum(wj * xj for wj, xj in zip(w, xi)) for xi in X]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    seq_path = find_input("rnn_sequences.json")
    print(f"Loading sequence records from: {seq_path}")
    records = load_json(seq_path)
    print(f"  {len(records):,} records")

    train_years, val_years, test_years = temporal_split(records)
    split_of_year = {}
    for y in train_years:
        split_of_year[y] = "train"
    for y in val_years:
        split_of_year[y] = "val"
    for y in test_years:
        split_of_year[y] = "test"
    print(f"Split -- train {min(train_years)}-{max(train_years)}  val {sorted(val_years)}  test {sorted(test_years)}")

    # ---- Season-level features, one weather cell per (state, season, year) ----
    cells = {}
    for r in records:
        key = (r["state"], r["season"], r["year"])
        if key not in cells:
            cells[key] = season_features(r)
    print(f"Weather cells (state, season, year): {len(cells):,}  "
          f"(the effective sample size -- every crop in a cell shares its weather)")

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
    n_no_clim = sum(1 for v in anomalies.values() if v is None)
    print(f"Climatology fitted on train-year cells for {len(clim)} (state, season) pairs; "
          f"{n_no_clim} cells had no usable climatology and are skipped")

    # ---- Rows ----
    rows = {"train": [], "val": [], "test": []}
    skipped = 0
    for r in records:
        an = anomalies.get((r["state"], r["season"], r["year"]))
        if an is None:
            skipped += 1
            continue
        rows[split_of_year[r["year"]]].append({
            "cluster": (r["state"], r["year"]),
            "season": r["season"],
            "y": 1 if r["risk"] == "At-Risk" else 0,
            "anom": an,
        })
    print(f"Rows used: train={len(rows['train']):,} val={len(rows['val']):,} "
          f"test={len(rows['test']):,}  (skipped {skipped:,} without climatology)")
    for sp in ("train", "val", "test"):
        for season in SEASONS:
            sub = [x for x in rows[sp] if x["season"] == season]
            share = sum(x["y"] for x in sub) / len(sub) if sub else float("nan")
            print(f"  {sp:<5} {season:<7} n={len(sub):>5,}  At-Risk share={share:.3f}")

    # ---- Univariate AUCs: pooled and per season ----
    def items_for(split, group, feature):
        out = []
        for x in rows[split]:
            if group != "ALL" and x["season"] != group:
                continue
            s = x["anom"].get(feature)
            if s is not None:
                out.append((x["cluster"], x["y"], s))
        return out

    def evaluate_feature(group, feature):
        aucs, test_items = {}, None
        for sp in ("train", "val", "test"):
            items = items_for(sp, group, feature)
            n_pos = sum(y for _, y, _ in items)
            if n_pos < MIN_CLASS_COUNT or (len(items) - n_pos) < MIN_CLASS_COUNT:
                return None
            aucs[sp] = roc_auc([y for _, y, _ in items], [s for _, _, s in items], 1)
            if sp == "test":
                test_items = items
        sign = 1 if aucs["train"] >= 0.5 else -1
        orient = (lambda a: a) if sign > 0 else (lambda a: 1.0 - a)
        return {
            "group": group, "feature": feature, "sign": sign,
            "train": orient(aucs["train"]), "val": orient(aucs["val"]), "test": orient(aucs["test"]),
            "test_items": [(c, y, sign * s) for c, y, s in test_items],
            "ci": None, "pass": False,
        }

    def attach_ci(res):
        if res["ci"] is None:
            res["ci"] = cluster_bootstrap_ci(res["test_items"], rng)
            res["pass"] = passes_rule(res["val"], res["test"], res["ci"][0])

    all_results = []
    for group in ["ALL"] + SEASONS:
        for f in FEATURE_NAMES:
            res = evaluate_feature(group, f)
            if res is not None:
                all_results.append(res)
    print(f"\nIndividual feature tests run: {len(all_results)}")

    def print_rows(title, results):
        print(f"\n{title}")
        print(f"  {'feature':<26}{'dir':>4}{'train':>8}{'val':>8}{'test':>8}   {'test 95% cluster CI':<22}{'pass'}")
        for res in results:
            attach_ci(res)
            lo, hi = res["ci"]
            ci_txt = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"
            print(f"  {res['feature']:<26}{('+' if res['sign'] > 0 else '-'):>4}"
                  f"{res['train']:>8.3f}{res['val']:>8.3f}{res['test']:>8.3f}   {ci_txt:<22}"
                  f"{'PASS' if res['pass'] else '-'}")

    print("\n" + "=" * 78)
    print("Univariate anomaly AUCs  (direction chosen on TRAIN, applied unchanged to val/test)")
    print("train column is in-sample by construction; val and test are the honest columns")
    print("dir '+' = higher anomaly -> more At-Risk;  '-' = higher anomaly -> less At-Risk")
    print("=" * 78)
    pooled = sorted((r for r in all_results if r["group"] == "ALL"), key=lambda r: -r["train"])
    print_rows("POOLED (all seasons, all crops) -- all features, ranked by train AUC", pooled)
    for season in SEASONS:
        per = sorted((r for r in all_results if r["group"] == season), key=lambda r: -r["train"])
        print_rows(f"{season.upper()} -- top 5 by train AUC", per[:5])

    # CI for any not-yet-evaluated feature that could still pass on its point estimates
    for res in all_results:
        if res["ci"] is None and res["test"] >= PASS_TEST_AUC and res["val"] >= PASS_VAL_AUC:
            attach_ci(res)
    n_pass = sum(1 for r in all_results if r["pass"])
    passing = [r for r in all_results if r["pass"]]
    print(f"\nIndividual features passing the rule: {n_pass} of {len(all_results)} tested")
    for r in passing:
        print(f"  PASS  {r['group']:<7} {r['feature']}  val={r['val']:.3f} test={r['test']:.3f} "
              f"CI=[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

    # ---- Combined per-season logistic baseline ----
    print("\n" + "=" * 78)
    print("Combined weather-anomaly baseline: logistic regression on all 25 anomalies, per season")
    print("(diagnostic only; crop-agnostic; AUC is raw -- no orientation applied)")
    print("=" * 78)
    combined = []
    for season in SEASONS:
        def mat(split):
            sub = [x for x in rows[split] if x["season"] == season]
            X = [[(x["anom"][f] if x["anom"][f] is not None else 0.0) for f in FEATURE_NAMES] for x in sub]
            return sub, X, [x["y"] for x in sub]

        tr_sub, Xtr, ytr = mat("train")
        va_sub, Xva, yva = mat("val")
        te_sub, Xte, yte = mat("test")
        if min(sum(ytr), len(ytr) - sum(ytr), sum(yva), len(yva) - sum(yva),
               sum(yte), len(yte) - sum(yte)) < MIN_CLASS_COUNT:
            print(f"  {season:<7} skipped: too few records/positives in a split")
            continue
        w, bias = fit_logistic(Xtr, ytr)
        auc_tr = roc_auc(ytr, logit_scores(Xtr, w, bias), 1)
        auc_va = roc_auc(yva, logit_scores(Xva, w, bias), 1)
        te_scores = logit_scores(Xte, w, bias)
        auc_te = roc_auc(yte, te_scores, 1)
        lo, hi = cluster_bootstrap_ci([(x["cluster"], y, s) for x, y, s in zip(te_sub, yte, te_scores)], rng)
        ok = passes_rule(auc_va, auc_te, lo)
        top = sorted(zip(FEATURE_NAMES, w), key=lambda t: -abs(t[1]))[:3]
        ci_txt = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"
        print(f"  {season:<7} train AUC {auc_tr:.3f} | val {auc_va:.3f} | test {auc_te:.3f} CI {ci_txt} "
              f"| {'PASS' if ok else '-'}")
        print(f"          top coefficients (standardized; + = higher anomaly -> more At-Risk): "
              + ", ".join(f"{n} {c:+.2f}" for n, c in top))
        combined.append({"season": season, "train_auc": auc_tr, "val_auc": auc_va, "test_auc": auc_te,
                         "test_ci95": [lo, hi], "pass": ok,
                         "top_coefficients": [{"feature": n, "weight": c} for n, c in top]})

    combined_pass = any(c["pass"] for c in combined)

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT (pre-registered rule: test AUC >= 0.55, val AUC >= 0.53, CI lower bound > 0.50)")
    print("=" * 78)
    if combined_pass or n_pass >= 3:
        verdict = "SIGNAL PRESENT"
        meaning = ("Weather anomalies predict At-Risk out of sample. The RNN's raw per-state input is "
                   "likely burying this under the seasonal cycle -> next: rebuild the RNN input as anomalies.")
    elif n_pass >= 1:
        verdict = "AMBIGUOUS"
        meaning = ("1-2 isolated passes among ~100 tests is within chance. Treat as no clear signal; "
                   "consider the spatial-resolution and feature fixes before more RNN work.")
    else:
        verdict = "NO USABLE SIGNAL"
        meaning = ("No anomaly or combination beats chance out of sample. The state-level weather point "
                   "carries no usable signal -> next: spatial resolution (district-level rows with local "
                   "weather) and more variables; more RNN tuning on this input would be wasted.")
    print(f"  Individual passes: {n_pass} of {len(all_results)}   |   Combined models passing: "
          f"{sum(1 for c in combined if c['pass'])} of {len(combined)}")
    print(f"  VERDICT: {verdict}")
    print(f"  {meaning}")

    out_path = seq_path.parent / "signal_diagnostic_results.json"
    serializable = []
    for res in all_results:
        serializable.append({
            "group": res["group"], "feature": res["feature"],
            "direction": "+" if res["sign"] > 0 else "-",
            "train_auc_oriented": res["train"], "val_auc_oriented": res["val"],
            "test_auc_oriented": res["test"],
            "test_ci95": list(res["ci"]) if res["ci"] is not None else None,
            "pass": res["pass"],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict,
            "rule": {"test_auc_min": PASS_TEST_AUC, "val_auc_min": PASS_VAL_AUC,
                     "ci_lower_bound_min": 0.5},
            "n_individual_tests": len(all_results), "n_individual_passes": n_pass,
            "combined_models": combined,
            "individual_features": serializable,
            "config": {"min_clim_years": MIN_CLIM_YEARS, "bootstrap_b": BOOTSTRAP_B,
                       "random_seed": RANDOM_SEED, "n_feature_names": len(FEATURE_NAMES)},
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()