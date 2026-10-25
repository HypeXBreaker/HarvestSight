"""
12_kharif_rnn.py

HarvestSight -- Option C: one more RNN run, restricted to Kharif.

WHY THIS EXISTS
---------------
11 found that a fixed, label-free moisture index ranks Kharif At-Risk records
on held-out years at test AUC 0.595 [0.515, 0.667] (top-20% lift x1.49), while
the pooled, all-season RNN (07 v3) reached only 0.529. The signal lives in
Kharif; pooling Rabi/Summer (where the same moisture direction is weak or
reversed) may have drowned it for the RNN. This script asks the one remaining
fair question: given ONLY Kharif records, can the same RNN extract held-out
signal -- and does it beat the transparent index?

WHAT IS UNCHANGED vs 07 v3
--------------------------
Everything in the model and training logic: Elman RNN, forward pass, BPTT,
plain SGD (batch size 1), hidden size 16, learning rate 0.005, 20 epochs,
per-state weather normalization, global soil normalization, sqrt-inverse-
frequency class weights, 30-dim input (6 weather + 3 soil + 21 crop one-hot,
the full shortlist vocabulary kept so the input width is identical),
validation-ROC-AUC checkpointing, test evaluated once at the checkpoint.
The temporal split is computed on ALL records exactly as in 07 (so the year
boundaries are identical: train 1997-2014, val 2015-16, test 2017-19) and
the Kharif rows are filtered out of each split afterwards. Normalization
statistics and class weights are fitted on the Kharif TRAIN rows only.

WHAT CHANGED
------------
  1. Row filter: Kharif only (SEASON_FILTER).
  2. Input: the weekly-anomaly file rnn_sequences_anomaly.json (09's output).
  3. 5 fixed seeds (SEEDS). One run on ~171 test positives is too noisy to
     read; the PRIMARY statistic is the MEAN test AUC across seeds, each seed
     using its own val-AUC checkpoint.
  4. The (state, year) cluster bootstrap resamples test clusters and averages
     the per-seed AUC inside every resample, giving a CI for that mean.

VALIDATION HOOK
---------------
Set SEASON_FILTER = None and SEEDS = [42] to run exactly 07 v3 on the anomaly
input. Epoch 1 must then print train_loss 0.6526 / val_auc 0.617 and say
MATCHES. This proves the training core here is identical to 07 v3. (Verdict
is skipped in that mode.)

PRE-REGISTERED VERDICT (fixed before running)
---------------------------------------------
On the Kharif TEST rows, primary = val-AUC checkpoint of each seed:
  PASS          mean test AUC >= 0.56 AND cluster-CI lower bound > 0.50
  FAIL          mean test AUC <  0.54
  INCONCLUSIVE  anything in between, or >= 0.56 with the CI including 0.50
  BEATS INDEX   only if PASS and mean test AUC >= 0.615 (index 0.595 + 0.02)
If PASS but not BEATS INDEX, the transparent moisture index stays the
headline and the RNN is reported as matching, not improving on, it.
Yardsticks (from 11, same 729 Kharif test rows): index test AUC 0.595
[0.515, 0.667], top-20% lift x1.49. Pooled RNN (07 v3 primary): 0.529.

CAVEATS (stated, not hidden)
----------------------------
  * ~171 test At-Risk records in ~80 (state, year) clusters: intervals are wide.
  * Val is 2 years (2015-16); picking the best of 20 epochs on it is noisy and
    its winning AUC is optimistic. Test is never used for any selection.
  * The 2017-19 test years have been looked at in earlier rounds (08, 11), and
    the index's feature choice came from 08, so the yardstick is not a clean
    out-of-sample number either.
  * The 5 seeds capture init/shuffle variance. The CI captures test-cluster
    variance only, not variance from a different training sample.
  * The comparison with the index is by point estimate against 11's printed
    numbers, not a paired test on identical resamples.
  * Test anomalies are shifted vs the train climatology (09: soil wetness
    +0.65 sigma, humidity +0.34 sigma), so inputs are partly out of
    distribution.

Input:
  rnn_sequences_anomaly.json   (09's output)
  crop_shortlist.json

Output:
  kharif_rnn_results.json   (written next to the input; per-seed summaries,
                             mean AUC, CI, verdict; also written after every
                             seed so a disconnect does not lose finished seeds)

Runtime: roughly (Kharif share ~ 0.53) x (07 v3 time) x (number of seeds).
A time estimate is printed after epoch 1 of the first seed. Stdlib only.
"""

import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEATHER_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
SOIL_PARAMS = ["phh2o", "soc", "clay"]
RISK_LABELS = ["Low", "At-Risk"]  # index 0 / 1
AT_RISK_IDX = RISK_LABELS.index("At-Risk")

HIDDEN_SIZE = 16
LEARNING_RATE = 0.005
EPOCHS = 20
N_TEST_YEARS = 3
TRAIN_FRACTION_OF_REMAINDER = 0.85

# --- 12-specific ---
SEASON_FILTER = "Kharif"          # None = all seasons (validation hook, see docstring)
SEEDS = [42, 43, 44, 45, 46]      # fixed in advance
INPUT_FILENAME = "rnn_sequences_anomaly.json"
EXPECTED_SPLIT_COUNTS_ALL = (7835, 980, 1446)   # train, val, test over ALL seasons
EXPECTED_KHARIF = {"records": 5445, "test": 729, "test_at_risk": 171}

TOP_FRACTION = 0.20
BOOTSTRAP_B = 300
PASS_AUC = 0.56
FAIL_AUC = 0.54
INDEX_TEST_AUC = 0.595
INDEX_TEST_CI = (0.515, 0.667)
INDEX_TOP_LIFT = 1.49
BEATS_INDEX_MARGIN = 0.02
POOLED_RNN_V3_TEST_AUC = 0.529

# Epoch-1 numbers from 07 v3 / v2 on the full anomaly input, seed 42.
BENCH_ANOMALY_EPOCH1 = {"train_loss": 0.6526, "val_auc": 0.617}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_crop_shortlist(path):
    raw = load_json(path)
    crops = raw["crops"] if isinstance(raw, dict) else raw
    return sorted(crops)  # deterministic one-hot order


def _project_root():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:  # running as a notebook cell (Colab): no __file__
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
# Manual matrix/vector ops (no numpy)  -- identical to 07 v3
# ---------------------------------------------------------------------------

def zeros_vec(n):
    return [0.0] * n


def zeros_mat(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def mat_vec(W, x):
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def tanh_vec(v):
    return [math.tanh(vi) for vi in v]


def outer(a, b):
    return [[ai * bi for bi in b] for ai in a]


def transpose(W):
    return [list(col) for col in zip(*W)]


def mat_add_inplace(dst, src):
    for i in range(len(dst)):
        for j in range(len(dst[i])):
            dst[i][j] += src[i][j]


def vec_add_inplace(dst, src):
    for i in range(len(dst)):
        dst[i] += src[i]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sgd_update_mat(W, dW, lr):
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] -= lr * dW[i][j]


def sgd_update_vec(v, dv, lr):
    for i in range(len(v)):
        v[i] -= lr * dv[i]


def random_matrix(rows, cols, rng):
    limit = 1.0 / math.sqrt(cols) if cols > 0 else 0.1
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# Temporal split  -- identical to 07 v3 (computed on ALL records)
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

    train = [r for r in records if r["year"] in train_years]
    val = [r for r in records if r["year"] in val_years]
    test = [r for r in records if r["year"] in test_years]
    return train, val, test, sorted(train_years), sorted(val_years), sorted(test_years)


# ---------------------------------------------------------------------------
# Normalization  -- identical to 07 v3 (fitted on whatever train rows are passed)
# ---------------------------------------------------------------------------

def compute_weather_normalization(train_records, all_states):
    per_state_values = {state: {p: [] for p in WEATHER_PARAMS} for state in all_states}
    global_values = {p: [] for p in WEATHER_PARAMS}

    for r in train_records:
        state = r["state"]
        sl = r["sequence_length"]
        bucket = per_state_values[state]
        for week in r["sequence"][:sl]:
            for i, p in enumerate(WEATHER_PARAMS):
                v = week[i]
                if v is not None:
                    bucket[p].append(v)
                    global_values[p].append(v)

    def mean_std(values):
        if not values:
            return 0.0, 1.0
        m = sum(values) / len(values)
        var = sum((v - m) ** 2 for v in values) / len(values)
        std = math.sqrt(var) if var > 0 else 1.0
        return m, std

    global_stats = {p: mean_std(global_values[p]) for p in WEATHER_PARAMS}

    stats = {}
    fallback_states = []
    for state, bucket in per_state_values.items():
        stats[state] = {}
        for p in WEATHER_PARAMS:
            if bucket[p]:
                stats[state][p] = mean_std(bucket[p])
            else:
                stats[state][p] = global_stats[p]
                fallback_states.append((state, p))

    return stats, global_stats, fallback_states


def compute_soil_normalization(train_records):
    values = {p: [] for p in SOIL_PARAMS}
    for r in train_records:
        for p in SOIL_PARAMS:
            values[p].append(r["soil"][p])

    stats = {}
    for p in SOIL_PARAMS:
        vals = values[p]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1.0
        stats[p] = (m, std)
    return stats


def normalize(value, mean, std):
    if value is None:
        return 0.0  # missing week -> normalized "at the mean" (design choice)
    return (value - mean) / std


def build_feature_sequence(record, weather_stats, soil_stats, crop_index):
    state = record["state"]
    sl = record["sequence_length"]
    state_stats = weather_stats.get(state)

    soil_norm = [normalize(record["soil"][p], *soil_stats[p]) for p in SOIL_PARAMS]

    crop_onehot = [0.0] * len(crop_index)
    crop_onehot[crop_index[record["crop"]]] = 1.0

    static_part = soil_norm + crop_onehot

    features = []
    for week in record["sequence"][:sl]:
        weather_norm = [
            normalize(week[i], *state_stats[p]) for i, p in enumerate(WEATHER_PARAMS)
        ]
        features.append(weather_norm + static_part)
    return features


# ---------------------------------------------------------------------------
# Forward / backward pass for one record  -- identical to 07 v3
# ---------------------------------------------------------------------------

def forward(features, params):
    W_xh, W_hh, b_h, W_out, b_out = params
    h = zeros_vec(HIDDEN_SIZE)
    hidden_states = [h]
    for x_t in features:
        z = vec_add(vec_add(mat_vec(W_xh, x_t), mat_vec(W_hh, h)), b_h)
        h = tanh_vec(z)
        hidden_states.append(h)
    logits = vec_add(mat_vec(W_out, hidden_states[-1]), b_out)
    probs = softmax(logits)
    return hidden_states, probs


def train_one_record(features, true_idx, params, class_weight, lr):
    W_xh, W_hh, b_h, W_out, b_out = params
    hidden_states, probs = forward(features, params)
    h_final = hidden_states[-1]

    weight = class_weight[true_idx]
    d_logits = [weight * (probs[i] - (1.0 if i == true_idx else 0.0)) for i in range(len(RISK_LABELS))]
    loss = -weight * math.log(max(probs[true_idx], 1e-12))

    dW_out = outer(d_logits, h_final)
    db_out = d_logits[:]
    dh_next = mat_vec(transpose(W_out), d_logits)

    dW_xh = zeros_mat(HIDDEN_SIZE, len(features[0]))
    dW_hh = zeros_mat(HIDDEN_SIZE, HIDDEN_SIZE)
    db_h = zeros_vec(HIDDEN_SIZE)
    W_hh_T = transpose(W_hh)

    for t in range(len(features) - 1, -1, -1):
        h_t = hidden_states[t + 1]
        h_prev = hidden_states[t]
        x_t = features[t]
        dtanh = [(1.0 - h_t[i] ** 2) * dh_next[i] for i in range(HIDDEN_SIZE)]
        vec_add_inplace(db_h, dtanh)
        mat_add_inplace(dW_xh, outer(dtanh, x_t))
        mat_add_inplace(dW_hh, outer(dtanh, h_prev))
        dh_next = mat_vec(W_hh_T, dtanh)

    sgd_update_mat(W_xh, dW_xh, lr)
    sgd_update_mat(W_hh, dW_hh, lr)
    sgd_update_vec(b_h, db_h, lr)
    sgd_update_mat(W_out, dW_out, lr)
    sgd_update_vec(b_out, db_out, lr)

    return loss


def predict_probs(features, params):
    _, probs = forward(features, params)
    return probs


# ---------------------------------------------------------------------------
# Metrics  -- roc_auc / top_fraction_precision identical to 07 v3 and 11
# ---------------------------------------------------------------------------

def roc_auc(true_labels, scores, positive_idx):
    """Rank-based ROC-AUC (Mann-Whitney U), ties counting half.
    Returns None if either class is absent."""
    n_pos = sum(1 for t in true_labels if t == positive_idx)
    n_neg = len(true_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, true_labels))  # ascending by score
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for a tie group
        for k in range(i, j + 1):
            if pairs[k][1] == positive_idx:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def top_fraction_precision(true_labels, scores, positive_idx, fraction=None):
    """Precision among the top `fraction` of records by score (ties broken by
    original order, so the result is deterministic). Threshold-free."""
    fraction = TOP_FRACTION if fraction is None else fraction
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


def _ci95(vals):
    vals = sorted(vals)
    last = len(vals) - 1
    return vals[int(0.025 * last)], vals[int(0.975 * last)]


def cluster_bootstrap(true_labels, scores, clusters, positive_idx, rng, b=None):
    """95% CIs for AUC, top-fraction precision and lift for ONE score vector,
    resampling whole (state, year) clusters. Same as 07 v3."""
    b = BOOTSTRAP_B if b is None else b
    by_cluster = defaultdict(list)
    for y, s, c in zip(true_labels, scores, clusters):
        by_cluster[c].append((y, s))
    ids = list(by_cluster)
    if len(ids) < 5:
        return None
    aucs, tops, lifts = [], [], []
    for _ in range(b):
        ys, ss = [], []
        for _ in range(len(ids)):
            for y, s in by_cluster[ids[rng.randrange(len(ids))]]:
                ys.append(y)
                ss.append(s)
        a = roc_auc(ys, ss, positive_idx)
        if a is None:
            continue
        tf = top_fraction_precision(ys, ss, positive_idx)
        aucs.append(a)
        tops.append(tf["precision"])
        lifts.append(tf["lift"])
    if len(aucs) < 20:
        return None
    return {"auc": _ci95(aucs), "top_precision": _ci95(tops), "lift": _ci95(lifts)}


def cluster_bootstrap_mean(true_labels, scores_by_seed, clusters, positive_idx, rng, b=None):
    """New in 12. 95% CIs for the MEAN (over seeds) test AUC and MEAN top-fraction
    lift. In every (state, year)-cluster resample the AUC/lift of each seed is
    computed on the same resampled rows and then averaged over seeds."""
    b = BOOTSTRAP_B if b is None else b
    members = defaultdict(list)
    for i, c in enumerate(clusters):
        members[c].append(i)
    ids = list(members)
    if len(ids) < 5:
        return None
    mean_aucs, mean_lifts = [], []
    for _ in range(b):
        idx = []
        for _ in range(len(ids)):
            idx.extend(members[ids[rng.randrange(len(ids))]])
        ys = [true_labels[i] for i in idx]
        aucs, lifts = [], []
        complete = True
        for sc in scores_by_seed:
            ss = [sc[i] for i in idx]
            a = roc_auc(ys, ss, positive_idx)
            if a is None:
                complete = False
                break
            aucs.append(a)
            lifts.append(top_fraction_precision(ys, ss, positive_idx)["lift"])
        if complete:
            mean_aucs.append(sum(aucs) / len(aucs))
            mean_lifts.append(sum(lifts) / len(lifts))
    if len(mean_aucs) < 20:
        return None
    return {"auc": _ci95(mean_aucs), "lift": _ci95(mean_lifts)}


def verdict_for(mean_auc, ci):
    """Pre-registered reading (see module docstring)."""
    if mean_auc is None:
        return "FAIL"
    if mean_auc >= PASS_AUC and ci is not None and ci[0] > 0.5:
        return "PASS"
    if mean_auc < FAIL_AUC:
        return "FAIL"
    return "INCONCLUSIVE"


def mean_sd(vals):
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return m, sd


# ---------------------------------------------------------------------------
# One seed: train 20 epochs, checkpoint on val AUC, evaluate test once
# ---------------------------------------------------------------------------

def run_seed(seed, ctx, seed_position):
    rng = random.Random(seed)   # init + epoch shuffles (same draw order as 07 v3)
    input_size = ctx["input_size"]
    train_prepped, val_prepped, test_prepped = ctx["train"], ctx["val"], ctx["test"]
    val_true, test_true = ctx["val_true"], ctx["test_true"]
    class_weight = ctx["class_weight"]
    pct = int(round(TOP_FRACTION * 100))

    W_xh = random_matrix(HIDDEN_SIZE, input_size, rng)
    W_hh = random_matrix(HIDDEN_SIZE, HIDDEN_SIZE, rng)
    b_h = zeros_vec(HIDDEN_SIZE)
    W_out = random_matrix(len(RISK_LABELS), HIDDEN_SIZE, rng)
    b_out = zeros_vec(len(RISK_LABELS))
    params = (W_xh, W_hh, b_h, W_out, b_out)

    best_val_auc, best_epoch, best_params = -1.0, -1, None
    t_seed = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        order = list(range(len(train_prepped)))
        rng.shuffle(order)
        total_loss = 0.0
        for idx in order:
            feats, label_idx = train_prepped[idx]
            total_loss += train_one_record(feats, label_idx, params, class_weight, LEARNING_RATE)
        avg_train_loss = total_loss / len(train_prepped)

        val_scores = [predict_probs(feats, params)[AT_RISK_IDX] for feats, _ in val_prepped]
        val_auc = roc_auc(val_true, val_scores, AT_RISK_IDX)
        val_top = top_fraction_precision(val_true, val_scores, AT_RISK_IDX)

        marker = ""
        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            best_params = copy.deepcopy(params)
            marker = "  <- checkpoint"
        auc_txt = f"{val_auc:.3f}" if val_auc is not None else "n/a"
        print(f"  seed {seed} epoch {epoch:>2}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
              f"val_auc={auc_txt}  val_top{pct}_prec={val_top['precision']:.3f}{marker}")

        if epoch == 1:
            if seed_position == 0:
                per_epoch = time.time() - t_epoch
                per_seed = per_epoch * EPOCHS
                print(f"  (time: {per_epoch:.0f}s/epoch -> about {per_seed / 60:.1f} min per seed, "
                      f"about {per_seed * len(SEEDS) / 60:.0f} min for {len(SEEDS)} seed(s))")
            if SEASON_FILTER is None and seed == 42:
                ok = (abs(avg_train_loss - BENCH_ANOMALY_EPOCH1["train_loss"]) < 0.002
                      and val_auc is not None
                      and abs(val_auc - BENCH_ANOMALY_EPOCH1["val_auc"]) < 0.01)
                print(f"  (validation hook: 07 v3 epoch 1 was train_loss "
                      f"{BENCH_ANOMALY_EPOCH1['train_loss']:.4f} / val_auc "
                      f"{BENCH_ANOMALY_EPOCH1['val_auc']:.3f} -- "
                      f"{'MATCHES, training core is identical to 07 v3' if ok else 'DIFFERS: check input file / Python version'})")

    if best_params is None:      # val AUC undefined (cannot happen with both classes in val)
        best_params, best_epoch = copy.deepcopy(params), EPOCHS

    out = {"seed": seed, "best_epoch": best_epoch, "best_val_auc": best_val_auc,
           "seconds": time.time() - t_seed}
    for tag, prm, ep in (("primary", best_params, best_epoch), ("reference", params, EPOCHS)):
        scores = [predict_probs(feats, prm)[AT_RISK_IDX] for feats, _ in test_prepped]
        out[tag] = {
            "epoch": ep,
            "test_auc": roc_auc(test_true, scores, AT_RISK_IDX),
            "top_fraction": top_fraction_precision(test_true, scores, AT_RISK_IDX),
            "scores": scores,
        }
    return out


def summarize_seed(res):
    """Seed result without the bulky score vectors (for JSON / printing)."""
    slim = {k: v for k, v in res.items() if k not in ("primary", "reference")}
    for tag in ("primary", "reference"):
        slim[tag] = {k: v for k, v in res[tag].items() if k != "scores"}
    return slim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    boot_rng = random.Random(1042)   # bootstrap only; never touches training

    seq_path = find_input(INPUT_FILENAME)
    shortlist_path = find_input("crop_shortlist.json")

    print(f"Loading sequence records from: {seq_path}")
    records = load_json(seq_path)
    kind = records[0].get("sequence_kind", "raw_weekly_weather") if records else "unknown"
    print(f"  {len(records):,} total sequence records  |  input kind: {kind}")
    if kind != "weekly_anomaly_z":
        raise ValueError("This script expects the weekly-ANOMALY file from 09 "
                         "(sequence_kind = weekly_anomaly_z), not the raw sequence file.")
    if "season" not in records[0]:
        raise ValueError("Records carry no 'season' field, so a Kharif filter is impossible.")

    crops = load_crop_shortlist(shortlist_path)
    crop_index = {c: i for i, c in enumerate(crops)}
    input_size = len(WEATHER_PARAMS) + len(SOIL_PARAMS) + len(crops)
    print(f"  {len(crops)} crops -> input size per timestep: {input_size}")

    # ---- Split on ALL records (identical year boundaries to 07), then filter ----
    train_all, val_all, test_all, train_years, val_years, test_years = temporal_split(records)
    print(f"\nSplit (all seasons) -- train {train_years[0]}-{train_years[-1]}  "
          f"val {val_years}  test {test_years}")
    counts_all = (len(train_all), len(val_all), len(test_all))
    print(f"  train={counts_all[0]:,}  val={counts_all[1]:,}  test={counts_all[2]:,}  "
          f"(07 expects {EXPECTED_SPLIT_COUNTS_ALL[0]:,} / {EXPECTED_SPLIT_COUNTS_ALL[1]:,} / "
          f"{EXPECTED_SPLIT_COUNTS_ALL[2]:,})")
    if counts_all != EXPECTED_SPLIT_COUNTS_ALL:
        print("  WARNING: split differs from 07's -- is this the right anomaly file?")

    def keep(split):
        if SEASON_FILTER is None:
            return split
        return [r for r in split if r["season"] == SEASON_FILTER]

    train, val, test = keep(train_all), keep(val_all), keep(test_all)
    scope = SEASON_FILTER if SEASON_FILTER is not None else "ALL seasons (validation-hook mode)"
    n_scope = len(train) + len(val) + len(test)
    test_pos = sum(1 for r in test if r["risk"] == "At-Risk")
    print(f"\nScope: {scope}")
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}  (scope total {n_scope:,})")
    print(f"  test At-Risk: {test_pos} of {len(test)} ({test_pos / len(test):.3f})")
    if SEASON_FILTER == "Kharif":
        ok = (n_scope == EXPECTED_KHARIF["records"] and len(test) == EXPECTED_KHARIF["test"]
              and test_pos == EXPECTED_KHARIF["test_at_risk"])
        print(f"  benchmark from 11: {EXPECTED_KHARIF['records']:,} Kharif records, test "
              f"{EXPECTED_KHARIF['test']} rows / {EXPECTED_KHARIF['test_at_risk']} At-Risk -- "
              f"{'MATCHES' if ok else 'DIFFERS: check the input file'}")

    def share(split):
        return sum(1 for r in split if r["risk"] == "At-Risk") / len(split) if split else float("nan")

    shares = {"train": share(train), "val": share(val), "test": share(test)}
    print(f"  At-Risk share -- train {shares['train']:.3f}  val {shares['val']:.3f}  test {shares['test']:.3f}")

    # ---- Normalization and class weights (fit on in-scope TRAIN only) ----
    all_states = sorted({r["state"] for r in records})
    weather_stats, _, fallback_states = compute_weather_normalization(train, all_states)
    if fallback_states:
        print(f"  note: {len(fallback_states)} (state, param) pairs had no in-scope train weeks "
              f"and use global stats (expected for states without Kharif train rows)")
    soil_stats = compute_soil_normalization(train)

    train_counts = Counter(r["risk"] for r in train)
    class_weight = [math.sqrt(len(train) / (len(RISK_LABELS) * train_counts[label]))
                    for label in RISK_LABELS]
    print(f"  train class counts: {dict(train_counts)}  weights: "
          f"{dict(zip(RISK_LABELS, (round(w, 3) for w in class_weight)))}")

    def prep(split_records):
        return [(build_feature_sequence(r, weather_stats, soil_stats, crop_index),
                 RISK_LABELS.index(r["risk"])) for r in split_records]

    ctx = {
        "input_size": input_size, "class_weight": class_weight,
        "train": prep(train), "val": prep(val), "test": prep(test),
    }
    ctx["val_true"] = [y for _, y in ctx["val"]]
    ctx["test_true"] = [y for _, y in ctx["test"]]
    test_clusters = [(r["state"], r["year"]) for r in test]
    print(f"  test clusters (state, year): {len(set(test_clusters))}")

    # ---- Train each seed ----
    out_path = seq_path.parent / "kharif_rnn_results.json"
    seed_results = []
    print("\n" + "=" * 70)
    print(f"Training {len(SEEDS)} seed(s); checkpoint = best validation ROC-AUC per seed")
    print("=" * 70)
    for pos, seed in enumerate(SEEDS):
        res = run_seed(seed, ctx, pos)
        seed_results.append(res)
        p = res["primary"]
        print(f"  --> seed {seed}: checkpoint epoch {res['best_epoch']} (val AUC {res['best_val_auc']:.3f})  "
              f"TEST AUC {p['test_auc']:.3f}  top-20% lift x{p['top_fraction']['lift']:.2f}  "
              f"[final-epoch ref AUC {res['reference']['test_auc']:.3f}]  ({res['seconds'] / 60:.1f} min)\n")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"complete": False, "scope": scope,
                       "seeds_done": [summarize_seed(r) for r in seed_results]}, f, indent=2)

    # ---- Aggregate ----
    test_true = ctx["test_true"]
    prim_aucs = [r["primary"]["test_auc"] for r in seed_results]
    ref_aucs = [r["reference"]["test_auc"] for r in seed_results]
    prim_lifts = [r["primary"]["top_fraction"]["lift"] for r in seed_results]
    m_auc, sd_auc = mean_sd(prim_aucs)
    m_ref, _ = mean_sd(ref_aucs)
    m_lift, _ = mean_sd(prim_lifts)
    boot = cluster_bootstrap_mean(test_true, [r["primary"]["scores"] for r in seed_results],
                                  test_clusters, AT_RISK_IDX, boot_rng)

    n_seeds = len(seed_results)
    ens_scores = [sum(r["primary"]["scores"][i] for r in seed_results) / n_seeds
                  for i in range(len(test_true))]
    ens_auc = roc_auc(test_true, ens_scores, AT_RISK_IDX)
    ens_top = top_fraction_precision(test_true, ens_scores, AT_RISK_IDX)
    ens_boot = cluster_bootstrap(test_true, ens_scores, test_clusters, AT_RISK_IDX, boot_rng)

    pct = int(round(TOP_FRACTION * 100))
    print("=" * 70)
    print(f"RESULT -- {scope}, TEST ({len(test_true)} rows, {sum(test_true)} At-Risk), "
          f"primary = val-AUC checkpoint per seed")
    print("=" * 70)
    print(f"  {'seed':<8}{'ckpt ep':>8}{'val AUC':>9}{'test AUC':>10}{'lift':>8}{'final-ep AUC':>14}")
    for r in seed_results:
        print(f"  {r['seed']:<8}{r['best_epoch']:>8}{r['best_val_auc']:>9.3f}"
              f"{r['primary']['test_auc']:>10.3f}{r['primary']['top_fraction']['lift']:>8.2f}"
              f"{r['reference']['test_auc']:>14.3f}")
    ci_txt = f" [{boot['auc'][0]:.3f}, {boot['auc'][1]:.3f}]" if boot else " [CI unavailable]"
    print(f"\n  MEAN test AUC over {n_seeds} seed(s): {m_auc:.3f}{ci_txt}   "
          f"(seed sd {sd_auc:.3f}, range {min(prim_aucs):.3f}-{max(prim_aucs):.3f})")
    lift_ci = f" [{boot['lift'][0]:.2f}, {boot['lift'][1]:.2f}]" if boot else ""
    print(f"  MEAN top-{pct}% lift: x{m_lift:.2f}{lift_ci}   (base rate {sum(test_true) / len(test_true):.3f})")
    print(f"  mean final-epoch (reference) test AUC: {m_ref:.3f}")
    e_ci = f" [{ens_boot['auc'][0]:.3f}, {ens_boot['auc'][1]:.3f}]" if ens_boot else ""
    print(f"  descriptive only -- seed-averaged score: AUC {ens_auc:.3f}{e_ci}, "
          f"top-{pct}% lift x{ens_top['lift']:.2f}")

    print("\n  Yardsticks (same Kharif test rows, from 11 / 07 v3):")
    print(f"    moisture index   test AUC {INDEX_TEST_AUC:.3f} "
          f"[{INDEX_TEST_CI[0]:.3f}, {INDEX_TEST_CI[1]:.3f}]   top-{pct}% lift x{INDEX_TOP_LIFT:.2f}")
    print(f"    pooled RNN (07 v3 primary, all seasons)   test AUC {POOLED_RNN_V3_TEST_AUC:.3f}")

    verdict = verdict_for(m_auc, boot["auc"] if boot else None)
    beats = verdict == "PASS" and m_auc >= INDEX_TEST_AUC + BEATS_INDEX_MARGIN

    print("\n" + "=" * 70)
    print("VERDICT (pre-registered)")
    print("=" * 70)
    if SEASON_FILTER is None:
        print("  Validation-hook mode (no season filter): verdict not applicable.")
        verdict, beats = "N/A", False
    else:
        print(f"  PASS          mean test AUC >= {PASS_AUC} and cluster-CI lower bound > 0.50")
        print(f"  FAIL          mean test AUC <  {FAIL_AUC}")
        print(f"  INCONCLUSIVE  otherwise")
        print(f"  BEATS INDEX   PASS and mean test AUC >= {INDEX_TEST_AUC + BEATS_INDEX_MARGIN:.3f}")
        print(f"  --> {verdict}" + ("  |  BEATS INDEX" if beats else ""))
        if verdict == "FAIL":
            print("  Even restricted to Kharif, the RNN extracts no held-out signal. The transparent "
                  "moisture index is the headline; the RNN is reported as tested and ruled out.")
        elif verdict == "INCONCLUSIVE":
            print("  Some signal, not reliably above chance. The index stays the headline; "
                  "report the RNN as a weak, unconfirmed lead.")
        elif beats:
            print("  The RNN beats the index on the same rows. Report both; confirm with year-blocked "
                  "CV before claiming the RNN is better.")
        else:
            print("  The RNN recovers held-out signal but does not improve on the index. The index "
                  "stays the headline (simpler, explainable, same skill).")
    if n_seeds != 5 and SEASON_FILTER is not None:
        print(f"  NOTE: {n_seeds} seed(s) were run; the pre-registration specifies 5.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "complete": True, "scope": scope, "input_kind": kind,
            "hyperparameters": {"hidden_size": HIDDEN_SIZE, "learning_rate": LEARNING_RATE,
                                "epochs": EPOCHS, "seeds": SEEDS},
            "split": {"train_years": train_years, "val_years": val_years, "test_years": test_years,
                      "train": len(train), "val": len(val), "test": len(test),
                      "at_risk_share": shares},
            "seeds": [summarize_seed(r) for r in seed_results],
            "mean_test_auc": m_auc, "mean_test_auc_ci95": boot["auc"] if boot else None,
            "mean_top_lift": m_lift, "mean_top_lift_ci95": boot["lift"] if boot else None,
            "mean_reference_test_auc": m_ref,
            "seed_averaged_score": {"auc": ens_auc, "top_fraction": ens_top,
                                    "cluster_bootstrap_ci95": ens_boot},
            "yardsticks": {"index_test_auc": INDEX_TEST_AUC, "index_test_ci": INDEX_TEST_CI,
                           "index_top_lift": INDEX_TOP_LIFT,
                           "pooled_rnn_v3_test_auc": POOLED_RNN_V3_TEST_AUC},
            "verdict": verdict, "beats_index": beats,
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()