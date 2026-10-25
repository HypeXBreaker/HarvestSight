"""
07_train_model.py   (v3 -- validation-AUC checkpointing, ranking metrics, cluster-bootstrap CIs)

HarvestSight -- Phase 6/7: Model Training.

Hand-coded, plain (Elman) RNN -- no ML libraries, no numpy. Trained one
record at a time (batch size 1, plain SGD), which sidesteps padding/masking
entirely: each record's RNN only steps through its own real
`sequence_length` weeks, never the zero-padded tail added for storage.

    h_t = tanh(W_xh . x_t + W_hh . h_(t-1) + b_h)
    logits = W_out . h_T + b_out         (T = final real timestep)

Per-timestep input: 6 weather + 3 soil + 21 crop one-hot = 30 features,
concatenated at every timestep (soil and crop identity are static per
record, so the same 24 values repeat every step; only the 6 weather values
change week to week).

WHY v3 EXISTS
-------------
v2 trained on the weekly-anomaly input (09_build_anomaly_sequences.py) and
failed on test (ROC-AUC 0.486, precision lift x0.86). But the per-epoch
validation AUC told a different story: 0.617 / 0.637 / 0.607 in epochs 1-3,
about 0.59 on average over epochs 1-13 (vs about 0.51 for the raw-input
run), then a decay to 0.515 by epoch 20 while train loss kept falling
(0.653 -> 0.601). Epoch-1 train loss is roughly what a constant predictor
scores, so the early model had barely fit anything. The model that got
evaluated was the overfit END of that curve, not the one that had learned
the signal.

Checkpointing was tried earlier and dropped, but it selected on F1 at a
fixed 0.5 threshold, which swings wildly here (val At-Risk F1 ran 0.043 to
0.365 between epochs) because the model's flag rate crosses the threshold.
AUC has no threshold and shows a clear trend, so it is a far steadier
selection criterion.

WHAT CHANGED IN v3 (stopping and evaluation ONLY)
-------------------------------------------------
  1. CHECKPOINT ON VALIDATION ROC-AUC. All 20 epochs still run (the full
     curve is printed), but the weights from the epoch with the best val
     AUC are kept. Test is evaluated ONCE at that checkpoint (PRIMARY).
     The final-epoch weights are evaluated too, as a REFERENCE for
     comparison -- the primary result is fixed in advance as the
     val-AUC checkpoint, so this is not a second chance to pick a winner.
  2. TOP-20% PRECISION (threshold-free). Precision among the 20% of records
     the model scores riskiest, against the base rate, plus lift. This is
     how an early-warning list would actually be used, and it doesn't
     depend on where a 0.5 cutoff falls.
  3. (STATE, YEAR) CLUSTER-BOOTSTRAP CONFIDENCE INTERVALS for test AUC and
     top-20% precision/lift. v2's interval treated records as independent;
     every crop in a (state, year) shares weather, so it was optimistic.
  4. INPUT FILE: looks for rnn_sequences_anomaly.json first, then
     rnn_sequences.json, and prints which one it used and what kind of
     sequence it holds. No manual renaming needed. The output model file
     gets an "_anomaly" suffix for anomaly input so it can't overwrite a
     raw-input model.

UNCHANGED (verified by source comparison against v2): the forward pass,
backprop through time, SGD update, per-state weather normalization, soil
normalization, class weights, split, feature builder, hyperparameters
(HIDDEN_SIZE=16, LEARNING_RATE=0.005, EPOCHS=20), and seed 42. Training on
the same input therefore reproduces v2's per-epoch train loss and val
metrics; an epoch-1 reproducibility check is printed.

PRE-REGISTERED READING OF THE RESULT (fixed before running)
-----------------------------------------------------------
Judged on the PRIMARY (val-AUC checkpoint), on TEST:
  PASS  test AUC >= 0.55, cluster-CI lower bound > 0.50, top-20% lift > 1.0
  WEAK  test AUC >= 0.54 and top-20% lift > 1.0, but the CI includes 0.5
  FAIL  otherwise
Context from the 08 diagnostic: the best pooled single anomaly feature
scored 0.555 on test (0.594 in Kharif), and 08's per-season combined
logistic models scored 0.52-0.55. A realistic best case here is a test AUC
around 0.52-0.57.
  If FAIL: the RNN does not extract held-out signal from this input. Next
  experiments are dropping soil/crop identity (soil identifies the state,
  which invites memorization) and a transparent per-season moisture
  baseline as the benchmark.

CAVEATS (stated, not hidden)
----------------------------
  * Val is only 2 years (2015-16): picking the best of 20 epochs on it is
    noisy, and the winning val AUC is optimistic (winner's curse). Test is
    unbiased with respect to that choice, because test is never used to
    select anything -- but the 2017-19 test years HAVE been looked at in
    earlier rounds of this project, and only 3 years is a thin test.
  * Test-year anomalies are shifted (09's drift check: root-zone wetness
    +0.65 sigma, humidity +0.34 sigma vs the train climatology), so the
    model sees inputs partly outside its training distribution.
  * Year-blocked cross-validation across all 23 years would give steadier
    numbers; it is not done here.

Input:
  rnn_sequences_anomaly.json (or rnn_sequences.json)
  crop_shortlist.json

Output:
  harvestsight_rnn_anomaly.json  (harvestsight_rnn.json for raw input)

Stdlib only.
"""

import copy
import json
import math
import random
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
RANDOM_SEED = 42

# Expected sizes after 05 v2 (MIN_YEARS=8) + 06.
EXPECTED_SPLIT_COUNTS = (7835, 980, 1446)   # train, val, test

# v3 additions
INPUT_FILENAMES = ["rnn_sequences_anomaly.json", "rnn_sequences.json"]  # first match wins
TOP_FRACTION = 0.20
BOOTSTRAP_B = 300
PASS_TEST_AUC = 0.55
WEAK_TEST_AUC = 0.54

# Epoch-1 numbers from the v2 run on the anomaly input (training logic is identical).
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


def find_first_input(filenames):
    """First filename in the list that can be found wins."""
    last_error = None
    for name in filenames:
        try:
            return find_input(name)
        except FileNotFoundError as e:
            last_error = e
    raise FileNotFoundError(f"None of {filenames} found. Last search:\n{last_error}")


def model_output_dir(sequences_path):
    root = _project_root()
    if root is not None and (root / "cleaned_data").is_dir():
        return root / "model"
    return sequences_path.parent


# ---------------------------------------------------------------------------
# Manual matrix/vector ops (no numpy)
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
# Temporal split
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
# Normalization
# ---------------------------------------------------------------------------

def compute_weather_normalization(train_records, all_states):
    """Per-state mean/std for each weather param, from train-period real
    (unpadded) weeks only. Falls back to the global train-set stat for any
    state with zero usable train weeks -- covers both a state having some
    train records but missing values for one param, AND a state having no
    train records at all (e.g. all its records fell in val/test years).
    `all_states` is every state seen anywhere in the full dataset, so
    every state gets a bucket up front and the fallback applies uniformly."""
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
    """Global mean/std per soil param, from train-set records only
    (confirmed choice: soil stays globally normalized)."""
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


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def build_feature_sequence(record, weather_stats, soil_stats, crop_index):
    """Returns a list of `sequence_length` vectors, each 30-dim: 6 weather
    (per-state normalized) + 3 soil (globally normalized) + 21 crop one-hot
    (static features repeated at every timestep)."""
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
# Forward / backward pass for one record (UNCHANGED from v2)
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


def argmax(values):
    return values.index(max(values))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(true_labels, pred_labels):
    n = len(RISK_LABELS)
    confusion = zeros_mat(n, n)
    for t, p in zip(true_labels, pred_labels):
        confusion[t][p] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(n):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(n) if r != c)
        fn = sum(confusion[c][r] for r in range(n) if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = sum(confusion[c][c] for c in range(n)) / max(1, len(true_labels))
    macro_f1 = sum(f1s) / n
    return {
        "confusion_matrix": confusion,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def roc_auc(true_labels, scores, positive_idx):
    """Rank-based ROC-AUC (Mann-Whitney U): the probability that a randomly
    chosen positive record gets a higher score than a randomly chosen
    negative one, ties counting half. 0.5 = no better than chance.
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


def chance_comparison(true_labels, pred_labels, positive_idx):
    """What the model scored on the positive class at its 0.5 decision rule,
    next to a random guesser flagging at the SAME rate. Chance precision
    equals the base rate; chance recall equals the flag rate."""
    n = len(true_labels)
    n_pos = sum(1 for t in true_labels if t == positive_idx)
    n_flagged = sum(1 for p in pred_labels if p == positive_idx)
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == positive_idx and p == positive_idx)

    base_rate = n_pos / n if n else 0.0
    flag_rate = n_flagged / n if n else 0.0
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    chance_f1 = (2 * base_rate * flag_rate / (base_rate + flag_rate)
                 if (base_rate + flag_rate) else 0.0)
    return {
        "n": n, "n_positive": n_pos, "n_flagged": n_flagged,
        "base_rate": base_rate, "flag_rate": flag_rate,
        "precision": precision, "recall": recall, "f1": f1,
        "chance_precision": base_rate, "chance_recall": flag_rate, "chance_f1": chance_f1,
        "precision_lift": (precision / base_rate) if base_rate else 0.0,
    }


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


def cluster_bootstrap(true_labels, scores, clusters, positive_idx, rng, b=BOOTSTRAP_B):
    """95% CIs for AUC, top-fraction precision and lift, resampling whole
    (state, year) CLUSTERS. Every crop in a (state, season, year) shares the
    same weather, so records are not independent; resampling clusters gives
    wider, more honest intervals than resampling records.
    Returns {"auc": (lo, hi), "top_precision": (lo, hi), "lift": (lo, hi)} or None."""
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

    def ci(vals):
        vals = sorted(vals)
        last = len(vals) - 1
        return vals[int(0.025 * last)], vals[int(0.975 * last)]

    return {"auc": ci(aucs), "top_precision": ci(tops), "lift": ci(lifts)}


def skill_report(title, true_labels, probs, clusters, boot_rng):
    """Prints and returns the At-Risk skill block: threshold-free ranking
    metrics first (AUC, top-20% precision), then the 0.5-threshold view."""
    scores = [p[AT_RISK_IDX] for p in probs]
    preds = [argmax(p) for p in probs]
    cc = chance_comparison(true_labels, preds, AT_RISK_IDX)
    auc = roc_auc(true_labels, scores, AT_RISK_IDX)
    top = top_fraction_precision(true_labels, scores, AT_RISK_IDX)
    boot = cluster_bootstrap(true_labels, scores, clusters, AT_RISK_IDX, boot_rng)

    pct = int(round(TOP_FRACTION * 100))
    print(f"  [{title}]")
    print(f"    base rate (true At-Risk share): {cc['base_rate']:.3f}")
    if auc is not None:
        ci_txt = f"  95% cluster-CI [{boot['auc'][0]:.3f}, {boot['auc'][1]:.3f}]" if boot else ""
        print(f"    ROC-AUC            {auc:.3f}{ci_txt}   (0.500 = chance)")
    lift_ci = f"  lift 95% cluster-CI [{boot['lift'][0]:.2f}, {boot['lift'][1]:.2f}]" if boot else ""
    print(f"    top-{pct}% precision  {top['precision']:.3f}  vs base rate {top['base_rate']:.3f}  "
          f"(lift x{top['lift']:.2f}){lift_ci}")
    print(f"    at the 0.5 rule: flagged {cc['flag_rate']:.3f} of records | "
          f"precision {cc['precision']:.3f} (chance {cc['chance_precision']:.3f}) | "
          f"recall {cc['recall']:.3f} (chance {cc['chance_recall']:.3f}) | "
          f"F1 {cc['f1']:.3f} (chance {cc['chance_f1']:.3f})")
    return {"chance_comparison": cc, "roc_auc": auc, "top_fraction": top,
            "cluster_bootstrap_ci95": boot}


def verdict_for(report):
    """Pre-registered reading (see module docstring)."""
    auc = report["roc_auc"]
    boot = report["cluster_bootstrap_ci95"]
    lift = report["top_fraction"]["lift"]
    if auc is None or boot is None:
        return "FAIL"
    if auc >= PASS_TEST_AUC and boot["auc"][0] > 0.5 and lift > 1.0:
        return "PASS"
    if auc >= WEAK_TEST_AUC and lift > 1.0:
        return "WEAK"
    return "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)            # training: init + epoch shuffles (same draws as v2)
    boot_rng = random.Random(RANDOM_SEED + 1)   # bootstrap only; never touches training

    sequences_path = find_first_input(INPUT_FILENAMES)
    shortlist_path = find_input("crop_shortlist.json")

    print(f"Loading sequence records from: {sequences_path}")
    records = load_json(sequences_path)
    kind = records[0].get("sequence_kind", "raw_weekly_weather") if records else "unknown"
    print(f"  {len(records):,} total sequence records  |  input kind: {kind}")

    print(f"Loading crop shortlist from: {shortlist_path}")
    crops = load_crop_shortlist(shortlist_path)
    crop_index = {c: i for i, c in enumerate(crops)}
    print(f"  {len(crops)} crops -> one-hot width {len(crops)}")

    input_size = len(WEATHER_PARAMS) + len(SOIL_PARAMS) + len(crops)
    print(f"Input size per timestep: {len(WEATHER_PARAMS)} weather + {len(SOIL_PARAMS)} soil "
          f"+ {len(crops)} crop one-hot = {input_size}")

    # ---- Temporal split ----
    train, val, test, train_years, val_years, test_years = temporal_split(records)
    print()
    print(f"Split -- test years (most recent {N_TEST_YEARS}): {test_years}")
    print(f"Split -- val years:   {val_years}")
    print(f"Split -- train years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    actual_counts = (len(train), len(val), len(test))
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        print(f"  WARNING: expected train/val/test = {EXPECTED_SPLIT_COUNTS[0]:,} / "
              f"{EXPECTED_SPLIT_COUNTS[1]:,} / {EXPECTED_SPLIT_COUNTS[2]:,}. Got a different split -- "
              f"is this the right (relabeled) sequence file?")

    def at_risk_share(split):
        return sum(1 for r in split if r["risk"] == "At-Risk") / len(split) if split else float("nan")

    shares = {"train": at_risk_share(train), "val": at_risk_share(val), "test": at_risk_share(test)}
    print(f"At-Risk share -- train: {shares['train']:.3f}  val: {shares['val']:.3f}  "
          f"test: {shares['test']:.3f}")

    # ---- Normalization (fit on train only) ----
    all_states = sorted({r["state"] for r in records})
    weather_stats, weather_global_stats, fallback_states = compute_weather_normalization(train, all_states)
    if fallback_states:
        print(f"  WARNING: {len(fallback_states)} (state, param) pairs had zero train-period "
              f"weeks, fell back to global stats: {fallback_states}")
    soil_stats = compute_soil_normalization(train)
    print(f"Weather normalization: per-state ({len(weather_stats)} states fitted from train)")
    print(f"Soil normalization: global -- " +
          ", ".join(f"{p} mean={m:.2f} std={s:.2f}" for p, (m, s) in soil_stats.items()))

    # ---- Class weights (sqrt inverse frequency, from train only) ----
    train_counts = Counter(r["risk"] for r in train)
    n_train = len(train)
    class_weight = []
    for label in RISK_LABELS:
        count = train_counts[label]
        raw = n_train / (len(RISK_LABELS) * count)
        class_weight.append(math.sqrt(raw))
    print(f"Train class counts: {dict(train_counts)}")
    print(f"Class weights (sqrt inverse-freq): "
          f"{dict(zip(RISK_LABELS, (round(w, 3) for w in class_weight)))}")

    # ---- Pre-build feature sequences once ----
    def prep(split_records):
        prepped = []
        for r in split_records:
            feats = build_feature_sequence(r, weather_stats, soil_stats, crop_index)
            label_idx = RISK_LABELS.index(r["risk"])
            prepped.append((feats, label_idx))
        return prepped

    train_prepped = prep(train)
    val_prepped = prep(val)
    test_prepped = prep(test)
    val_true = [y for _, y in val_prepped]
    test_true = [y for _, y in test_prepped]
    val_clusters = [(r["state"], r["year"]) for r in val]
    test_clusters = [(r["state"], r["year"]) for r in test]

    # ---- Initialize weights (same draw order as v2) ----
    W_xh = random_matrix(HIDDEN_SIZE, input_size, rng)
    W_hh = random_matrix(HIDDEN_SIZE, HIDDEN_SIZE, rng)
    b_h = zeros_vec(HIDDEN_SIZE)
    W_out = random_matrix(len(RISK_LABELS), HIDDEN_SIZE, rng)
    b_out = zeros_vec(len(RISK_LABELS))
    params = (W_xh, W_hh, b_h, W_out, b_out)

    # ---- Training loop (logic unchanged; adds val-AUC checkpointing) ----
    print()
    print("=" * 66)
    print("Training  (all epochs run; checkpoint = best validation ROC-AUC)")
    print("=" * 66)
    pct = int(round(TOP_FRACTION * 100))
    best_val_auc = -1.0
    best_epoch = -1
    best_params = None

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_prepped)))
        rng.shuffle(order)
        total_loss = 0.0
        for idx in order:
            feats, label_idx = train_prepped[idx]
            loss = train_one_record(feats, label_idx, params, class_weight, LEARNING_RATE)
            total_loss += loss
        avg_train_loss = total_loss / len(train_prepped)

        val_probs = [predict_probs(feats, params) for feats, _ in val_prepped]
        val_pred = [argmax(p) for p in val_probs]
        val_scores = [p[AT_RISK_IDX] for p in val_probs]
        val_metrics = compute_metrics(val_true, val_pred)
        val_auc = roc_auc(val_true, val_scores, AT_RISK_IDX)
        val_top = top_fraction_precision(val_true, val_scores, AT_RISK_IDX)

        marker = ""
        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_params = copy.deepcopy(params)
            marker = "  <- best val AUC so far, checkpointed"

        auc_txt = f"{val_auc:.3f}" if val_auc is not None else "n/a"
        print(f"Epoch {epoch:>2}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.3f}  val_macro_f1={val_metrics['macro_f1']:.3f}  "
              f"val_auc={auc_txt}  val_top{pct}_prec={val_top['precision']:.3f}{marker}")

        if epoch == 1 and kind == "weekly_anomaly_z":
            ok = (abs(avg_train_loss - BENCH_ANOMALY_EPOCH1["train_loss"]) < 0.002
                  and val_auc is not None
                  and abs(val_auc - BENCH_ANOMALY_EPOCH1["val_auc"]) < 0.01)
            print(f"  (reproducibility: v2 run on this input had epoch-1 train_loss "
                  f"{BENCH_ANOMALY_EPOCH1['train_loss']:.4f} / val_auc "
                  f"{BENCH_ANOMALY_EPOCH1['val_auc']:.3f} -- "
                  f"{'MATCHES, so training logic is unchanged' if ok else 'DIFFERS: check the input file / Python version'})")

    print(f"\nBest validation ROC-AUC: {best_val_auc:.3f} at epoch {best_epoch}/{EPOCHS}  "
          f"(final epoch {EPOCHS} is the reference)")

    final_params = params

    # ---- Evaluation: PRIMARY (val-AUC checkpoint) and REFERENCE (final epoch) ----
    results = {}
    for tag, prm, epoch_id in (("PRIMARY", best_params, best_epoch), ("REFERENCE", final_params, EPOCHS)):
        label = (f"{tag}: val-AUC checkpoint, epoch {epoch_id}" if tag == "PRIMARY"
                 else f"{tag}: final-epoch weights, epoch {epoch_id}")
        print()
        print("=" * 66)
        print(label)
        print("=" * 66)
        v_probs = [predict_probs(feats, prm) for feats, _ in val_prepped]
        t_probs = [predict_probs(feats, prm) for feats, _ in test_prepped]
        t_pred = [argmax(p) for p in t_probs]
        t_metrics = compute_metrics(test_true, t_pred)
        print(f"Test accuracy {t_metrics['accuracy']:.3f} | macro-F1 {t_metrics['macro_f1']:.3f}  "
              f"(always answering Low would score accuracy {1 - shares['test']:.3f})")
        print(f"Confusion matrix (rows=true, cols=pred, order={RISK_LABELS}):")
        for row in t_metrics["confusion_matrix"]:
            print(f"  {row}")
        print()
        val_rep = skill_report("val", val_true, v_probs, val_clusters, boot_rng)
        print()
        test_rep = skill_report("TEST", test_true, t_probs, test_clusters, boot_rng)
        results[tag] = {"epoch": epoch_id, "test_metrics": t_metrics,
                        "val_skill": val_rep, "test_skill": test_rep}

    # ---- Pre-registered verdict on the PRIMARY ----
    verdict = verdict_for(results["PRIMARY"]["test_skill"])
    print()
    print("=" * 66)
    print("VERDICT on the PRIMARY checkpoint (pre-registered)")
    print("=" * 66)
    print(f"  PASS  test AUC >= {PASS_TEST_AUC}, cluster-CI lower bound > 0.50, top-{pct}% lift > 1.0")
    print(f"  WEAK  test AUC >= {WEAK_TEST_AUC} and top-{pct}% lift > 1.0, CI includes 0.5")
    print(f"  FAIL  otherwise")
    print(f"  --> {verdict}")
    if verdict == "FAIL":
        print("  The RNN does not extract held-out signal from this input. Next: drop soil/crop identity "
              "(memorization), and benchmark against a transparent per-season moisture baseline.")
    elif verdict == "WEAK":
        print("  A weak, not-yet-reliable signal. Treat as a lead; confirm with year-blocked CV before relying on it.")
    else:
        print("  The RNN recovers a held-out signal. Confirm with year-blocked CV; compare to the transparent baseline.")

    # ---- Save model (primary checkpoint) ----
    model_dir = model_output_dir(sequences_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_anomaly" if kind == "weekly_anomaly_z" else ""
    model_output_path = model_dir / f"harvestsight_rnn{suffix}.json"
    W_xh_b, W_hh_b, b_h_b, W_out_b, b_out_b = best_params
    model_out = {
        "weights": {"W_xh": W_xh_b, "W_hh": W_hh_b, "b_h": b_h_b, "W_out": W_out_b, "b_out": b_out_b},
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE, "input_size": input_size, "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS, "num_classes": len(RISK_LABELS), "momentum": False,
            "checkpoint_selection": f"best_val_roc_auc (epoch {best_epoch} of {EPOCHS})",
            "static_concat": "every_timestep", "random_seed": RANDOM_SEED,
        },
        "input_kind": kind,
        "label_scheme": "detrended_residual_v2 (per state/crop/season linear trend, z < -0.5 = At-Risk)",
        "risk_labels": RISK_LABELS,
        "crop_vocabulary": crops,
        "class_weights": dict(zip(RISK_LABELS, class_weight)),
        "normalization": {
            "weather_scope": "per_state",
            "weather": {state: {p: list(stat) for p, stat in stats.items()}
                        for state, stats in weather_stats.items()},
            "weather_global_fallback": {p: list(stat) for p, stat in weather_global_stats.items()},
            "soil_scope": "global",
            "soil": {p: list(stat) for p, stat in soil_stats.items()},
        },
        "split": {
            "test_years": test_years, "val_years": val_years, "train_years": train_years,
            "train_count": len(train), "val_count": len(val), "test_count": len(test),
            "at_risk_share": shares,
        },
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "verdict_primary": verdict,
        "primary": results["PRIMARY"],
        "reference_final_epoch": results["REFERENCE"],
    }
    with open(model_output_path, "w", encoding="utf-8") as f:
        json.dump(model_out, f, indent=2)

    print()
    print(f"Model (primary checkpoint) written to: {model_output_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()