"""
10_ablation_study.py

HarvestSight -- Phase 6 ablation: is the RNN using WEATHER, or memorizing
WHICH STATE AND CROP it's looking at?

WHY THIS EXISTS
---------------
07 v3 (val-AUC checkpointing, the anomaly input) still FAILED the
pre-registered test: val AUC 0.637 but test AUC only 0.529 [0.477, 0.579].
Four things have now been tried and ruled out one at a time: raw input,
class weighting, architecture, label drift. The anomaly input plus honest
checkpointing was the strongest remaining idea and it still didn't hold up
out of sample.

One explanation that hasn't been tested: 24 of the model's 30 input
features per timestep are soil (3) and crop one-hot (21) -- STATIC,
repeated at every week, identical for every record from the same state and
crop. Soil in particular is close to a state fingerprint: with 28 states
and only 3 soil numbers, the model may be learning "this state's baseline
risk" from soil+crop identity and barely using the 6 weather numbers at
all. If that's what's happening, the val/test gap is just the model
overfitting to which STATES happened to have more At-Risk years in
1997-2016 versus 2017-19 -- not a weather signal failing to generalize.

THE TEST
--------
Train the IDENTICAL model (same architecture, split, class weights,
checkpointing, seed) on FOUR input variants, changing only what's fed in:

  A. full        6 weather + 3 soil + 21 crop one-hot   (= 07 v3, baseline)
  B. weather_only    6 weather only (static features zeroed out)
  C. static_only     3 soil + 21 crop one-hot only (weather zeroed out)
  D. shuffled_weather  static features real; weather shuffled WITHIN each
                        (state, season) across train-year cells only (val/
                        test weather is untouched) -- this breaks any real
                        year-to-year weather signal while leaving the
                        state/season climate level and the static features
                        completely intact, so it isolates "can the model
                        use genuine weather variation" from "can it use
                        which state/season this is".

READING THE RESULT (fixed before running)
------------------------------------------
  * If C (static_only) matches or beats A on test AUC: soil/crop identity
    is doing the work, not weather. CONFIRMS the memorization concern.
  * If B (weather_only) is close to A: weather alone carries what signal
    there is; static features were not required for the AUC achieved.
  * If D (shuffled_weather) drops to ~chance while A does not: A's result
    depends on real year-to-year weather variation (not just mean climate
    level), which is the one outcome that would argue for continuing to
    refine the RNN on weather.
  * If D performs SIMILARLY to A: A is not using real weather variation
    either -- whatever AUC A has comes from the static features or from
    the state/season mean level alone, matching C's finding.
  Both A and C passing to a similar degree, with D close to A, points the
  same direction C does.

  Every variant is also judged on its own against the SAME pre-registered
  bar as 07 v3 (test AUC >= 0.55, cluster-CI lower bound > 0.50, top-20%
  lift > 1.0). This determines what, if anything, in this pipeline is
  usable -- not just which variant is "best among four failures".

WHAT STAYS FIXED SO THE COMPARISON IS CLEAN
--------------------------------------------
Architecture, HIDDEN_SIZE=16, LEARNING_RATE=0.005, EPOCHS=20, split, class
weights (recomputed per variant from that variant's own train labels --
labels don't change, so weights are identical across variants in
practice), seed 42 (each variant gets an independent RNG stream seeded
from a per-variant offset of the same base seed, so init/shuffle draws
differ across variants but are reproducible), checkpoint = best val
ROC-AUC, same metrics and cluster-bootstrap CIs as 07 v3.

Runtime: ~4x a single 07 v3 run (4 variants x 20 epochs). On the same
input used for 07 v3, that's the anomaly file; this script auto-detects
which one is present.

Input:
  rnn_sequences_anomaly.json (or rnn_sequences.json)
  crop_shortlist.json

Output:
  ablation_results.json

Stdlib only.
"""

import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (identical to 07 v3 unless noted)
# ---------------------------------------------------------------------------

WEATHER_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"]
SOIL_PARAMS = ["phh2o", "soc", "clay"]
RISK_LABELS = ["Low", "At-Risk"]
AT_RISK_IDX = RISK_LABELS.index("At-Risk")

HIDDEN_SIZE = 16
LEARNING_RATE = 0.005
EPOCHS = 20
N_TEST_YEARS = 3
TRAIN_FRACTION_OF_REMAINDER = 0.85
RANDOM_SEED = 42

EXPECTED_SPLIT_COUNTS = (7835, 980, 1446)

INPUT_FILENAMES = ["rnn_sequences_anomaly.json", "rnn_sequences.json"]
TOP_FRACTION = 0.20
BOOTSTRAP_B = 300
PASS_TEST_AUC = 0.55
WEAK_TEST_AUC = 0.54

VARIANTS = ["full", "weather_only", "static_only", "shuffled_weather"]
VARIANT_SEED_OFFSET = {"full": 0, "weather_only": 1, "static_only": 2, "shuffled_weather": 3}


# ---------------------------------------------------------------------------
# I/O helpers (identical to 07 v3)
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_crop_shortlist(path):
    raw = load_json(path)
    crops = raw["crops"] if isinstance(raw, dict) else raw
    return sorted(crops)


def _project_root():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
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
    last_error = None
    for name in filenames:
        try:
            return find_input(name)
        except FileNotFoundError as e:
            last_error = e
    raise FileNotFoundError(f"None of {filenames} found. Last search:\n{last_error}")


def output_dir(sequences_path):
    root = _project_root()
    if root is not None and (root / "cleaned_data").is_dir():
        return root / "model"
    return sequences_path.parent


# ---------------------------------------------------------------------------
# Manual matrix/vector ops (identical to 07 v3)
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
# Split (identical to 07 v3)
# ---------------------------------------------------------------------------

def temporal_split(records):
    years = sorted({r["year"] for r in records})
    test_years = set(years[-N_TEST_YEARS:])
    remainder_years = [y for y in years if y not in test_years]

    remainder_by_year = Counter(r["year"] for r in records if r["year"] in remainder_years)
    remainder_total = sum(remainder_by_year.values())

    train_years = set()
    cumulative = 0
    for y in remainder_years:
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
# Normalization (identical to 07 v3)
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
        return 0.0
    return (value - mean) / std


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------

def build_shuffled_weather_map(records, train_years, rng):
    """For D: a permutation of TRAIN-year cells' weather WITHIN each
    (state, season), keyed by (state, season, year) -> source (state,
    season, year) to copy weather from. Val/test years are left untouched
    (identity mapping), so only train weather is scrambled."""
    cells_by_group = defaultdict(set)
    for r in records:
        if r["year"] in train_years:
            cells_by_group[(r["state"], r["season"])].add(r["year"])

    mapping = {}
    for (state, season), years in cells_by_group.items():
        years = sorted(years)
        shuffled = years[:]
        rng.shuffle(shuffled)
        for src_year, dst_year in zip(years, shuffled):
            mapping[(state, season, dst_year)] = (state, season, src_year)
    return mapping


def build_feature_sequence(record, weather_stats, soil_stats, crop_index, variant,
                            weather_source_lookup=None, shuffle_map=None):
    """variant in {"full", "weather_only", "static_only", "shuffled_weather"}.
    Every variant produces the SAME 30-dim width (6 weather + 3 soil + 21 crop
    one-hot): the excluded part is ZEROED, never dropped, so the architecture
    and parameter count are identical across variants and the comparison is
    fair. Only "full" and "shuffled_weather" carry both real weather and real
    static features at once.

    weather_source_lookup: (state, season, year) -> raw record, used only by
    shuffled_weather to pull a different cell's weather sequence."""
    state = record["state"]
    sl = record["sequence_length"]
    state_stats = weather_stats.get(state)

    include_weather = variant in ("full", "weather_only", "shuffled_weather")
    include_static = variant in ("full", "static_only", "shuffled_weather")

    if include_static:
        soil_norm = [normalize(record["soil"][p], *soil_stats[p]) for p in SOIL_PARAMS]
        crop_onehot = [0.0] * len(crop_index)
        crop_onehot[crop_index[record["crop"]]] = 1.0
    else:
        soil_norm = [0.0] * len(SOIL_PARAMS)
        crop_onehot = [0.0] * len(crop_index)
    static_part = soil_norm + crop_onehot

    source_seq = record["sequence"]
    if variant == "shuffled_weather" and shuffle_map is not None:
        key = (record["state"], record["season"], record["year"])
        if key in shuffle_map:
            src_key = shuffle_map[key]
            source_seq = weather_source_lookup[src_key]["sequence"]

    features = []
    for week in source_seq[:sl]:
        if include_weather:
            weather_norm = [normalize(week[i], *state_stats[p]) for i, p in enumerate(WEATHER_PARAMS)]
        else:
            weather_norm = [0.0] * len(WEATHER_PARAMS)
        features.append(weather_norm + static_part)
    return features


# ---------------------------------------------------------------------------
# Forward / backward pass (identical to 07 v3)
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

    input_size = len(features[0])
    dW_xh = zeros_mat(HIDDEN_SIZE, input_size)
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
# Metrics (identical to 07 v3)
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
    return {"confusion_matrix": confusion, "precision": precisions, "recall": recalls,
            "f1": f1s, "macro_f1": macro_f1, "accuracy": accuracy}


def roc_auc(true_labels, scores, positive_idx):
    n_pos = sum(1 for t in true_labels if t == positive_idx)
    n_neg = len(true_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, true_labels))
    rank_sum_pos = 0.0
    i, n = 0, len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == positive_idx:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def top_fraction_precision(true_labels, scores, positive_idx, fraction=TOP_FRACTION):
    n = len(true_labels)
    k = max(1, int(math.ceil(fraction * n)))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    hits = sum(1 for i in order[:k] if true_labels[i] == positive_idx)
    n_pos = sum(1 for t in true_labels if t == positive_idx)
    base_rate = n_pos / n
    precision = hits / k
    return {"k": k, "hits": hits, "precision": precision,
            "recall": (hits / n_pos) if n_pos else 0.0, "base_rate": base_rate,
            "lift": (precision / base_rate) if base_rate else 0.0}


def cluster_bootstrap(true_labels, scores, clusters, positive_idx, rng, b=BOOTSTRAP_B):
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


def skill_report(title, true_labels, probs, clusters, boot_rng, verbose=True):
    scores = [p[AT_RISK_IDX] for p in probs]
    preds = [argmax(p) for p in probs]
    auc = roc_auc(true_labels, scores, AT_RISK_IDX)
    top = top_fraction_precision(true_labels, scores, AT_RISK_IDX)
    boot = cluster_bootstrap(true_labels, scores, clusters, AT_RISK_IDX, boot_rng)
    if verbose:
        pct = int(round(TOP_FRACTION * 100))
        auc_txt = f"{auc:.3f}" if auc is not None else "n/a"
        ci_txt = f"  CI [{boot['auc'][0]:.3f}, {boot['auc'][1]:.3f}]" if boot else ""
        print(f"    [{title}] AUC {auc_txt}{ci_txt}  |  top-{pct}% precision {top['precision']:.3f} "
              f"(lift x{top['lift']:.2f})")
    return {"roc_auc": auc, "top_fraction": top, "cluster_bootstrap_ci95": boot}


def verdict_for(report):
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
# Train + evaluate one variant
# ---------------------------------------------------------------------------

def run_variant(variant, train, val, test, train_years, weather_stats, soil_stats,
                 crop_index, input_size_full, records_by_key, boot_rng):
    rng = random.Random(RANDOM_SEED + VARIANT_SEED_OFFSET[variant])

    shuffle_map = None
    if variant == "shuffled_weather":
        shuffle_map = build_shuffled_weather_map(train, train_years, rng)

    def prep(split_records):
        prepped = []
        for r in split_records:
            feats = build_feature_sequence(r, weather_stats, soil_stats, crop_index, variant,
                                            weather_source_lookup=records_by_key, shuffle_map=shuffle_map)
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

    train_counts = Counter(r["risk"] for r in train)
    n_train = len(train)
    class_weight = [math.sqrt(n_train / (len(RISK_LABELS) * train_counts[label])) for label in RISK_LABELS]

    input_size = input_size_full
    W_xh = random_matrix(HIDDEN_SIZE, input_size, rng)
    W_hh = random_matrix(HIDDEN_SIZE, HIDDEN_SIZE, rng)
    b_h = zeros_vec(HIDDEN_SIZE)
    W_out = random_matrix(len(RISK_LABELS), HIDDEN_SIZE, rng)
    b_out = zeros_vec(len(RISK_LABELS))
    params = (W_xh, W_hh, b_h, W_out, b_out)

    best_val_auc = -1.0
    best_epoch = -1
    best_params = None
    epoch_log = []

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_prepped)))
        rng.shuffle(order)
        total_loss = 0.0
        for idx in order:
            feats, label_idx = train_prepped[idx]
            total_loss += train_one_record(feats, label_idx, params, class_weight, LEARNING_RATE)
        avg_train_loss = total_loss / len(train_prepped)

        val_probs = [predict_probs(feats, params) for feats, _ in val_prepped]
        val_auc = roc_auc(val_true, [p[AT_RISK_IDX] for p in val_probs], AT_RISK_IDX)
        epoch_log.append({"epoch": epoch, "train_loss": avg_train_loss, "val_auc": val_auc})

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_params = copy.deepcopy(params)

        print(f"    epoch {epoch:>2}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
              f"val_auc={'n/a' if val_auc is None else f'{val_auc:.3f}'}"
              + ("  <- best" if epoch == best_epoch else ""))

    test_probs = [predict_probs(feats, best_params) for feats, _ in test_prepped]
    test_pred = [argmax(p) for p in test_probs]
    test_metrics = compute_metrics(test_true, test_pred)

    print(f"  Best val AUC {best_val_auc:.3f} at epoch {best_epoch}; test results at that checkpoint:")
    val_probs_best = [predict_probs(feats, best_params) for feats, _ in val_prepped]
    val_rep = skill_report("val", val_true, val_probs_best, val_clusters, boot_rng)
    test_rep = skill_report("test", test_true, test_probs, test_clusters, boot_rng)
    verdict = verdict_for(test_rep)
    print(f"    verdict: {verdict}")

    return {
        "variant": variant, "best_val_auc": best_val_auc, "best_epoch": best_epoch,
        "epoch_log": epoch_log, "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "val_skill": val_rep, "test_skill": test_rep, "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    boot_rng = random.Random(RANDOM_SEED + 100)  # bootstrap only; separate from all training RNGs

    sequences_path = find_first_input(INPUT_FILENAMES)
    shortlist_path = find_input("crop_shortlist.json")

    print(f"Loading sequence records from: {sequences_path}")
    records = load_json(sequences_path)
    kind = records[0].get("sequence_kind", "raw_weekly_weather") if records else "unknown"
    print(f"  {len(records):,} records  |  input kind: {kind}")

    crops = load_crop_shortlist(shortlist_path)
    crop_index = {c: i for i, c in enumerate(crops)}
    input_size_full = len(WEATHER_PARAMS) + len(SOIL_PARAMS) + len(crops)
    print(f"{len(crops)} crops  |  full input size {input_size_full} "
          f"({len(WEATHER_PARAMS)} weather + {len(SOIL_PARAMS)} soil + {len(crops)} crop one-hot)")

    train, val, test, train_years, val_years, test_years = temporal_split(records)
    print(f"Split: train={len(train):,} ({train_years[0]}-{train_years[-1]})  "
          f"val={len(val):,} ({val_years})  test={len(test):,} ({test_years})")
    if (len(train), len(val), len(test)) != EXPECTED_SPLIT_COUNTS:
        print(f"  WARNING: expected {EXPECTED_SPLIT_COUNTS} -- is this the right (relabeled) sequence file?")

    def at_risk_share(split):
        return sum(1 for r in split if r["risk"] == "At-Risk") / len(split) if split else float("nan")
    print(f"At-Risk share -- train {at_risk_share(train):.3f}  val {at_risk_share(val):.3f}  "
          f"test {at_risk_share(test):.3f}")

    all_states = sorted({r["state"] for r in records})
    weather_stats, _, fallback_states = compute_weather_normalization(train, all_states)
    if fallback_states:
        print(f"  WARNING: {len(fallback_states)} (state, param) pairs fell back to global weather stats")
    soil_stats = compute_soil_normalization(train)

    records_by_key = {}
    for r in records:
        records_by_key.setdefault((r["state"], r["season"], r["year"]), r)

    print()
    print("=" * 70)
    print(f"Running {len(VARIANTS)} variants, {EPOCHS} epochs each "
          f"(same architecture/split/weights/checkpointing as 07 v3)")
    print("=" * 70)

    all_results = {}
    for variant in VARIANTS:
        print(f"\n--- variant: {variant} ---")
        res = run_variant(variant, train, val, test, train_years, weather_stats, soil_stats,
                           crop_index, input_size_full, records_by_key, boot_rng)
        all_results[variant] = res

    # ---- Comparison table ----
    print()
    print("=" * 70)
    print("COMPARISON (test set, each variant's own best-val-AUC checkpoint)")
    print("=" * 70)
    pct = int(round(TOP_FRACTION * 100))
    print(f"  {'variant':<18}{'val AUC':>9}{'test AUC':>10}{'test CI':>18}{'top'+str(pct)+'% lift':>12}{'verdict':>9}")
    for variant in VARIANTS:
        r = all_results[variant]
        auc = r["test_skill"]["roc_auc"]
        boot = r["test_skill"]["cluster_bootstrap_ci95"]
        lift = r["test_skill"]["top_fraction"]["lift"]
        auc_txt = f"{auc:.3f}" if auc is not None else "n/a"
        ci_txt = f"[{boot['auc'][0]:.3f},{boot['auc'][1]:.3f}]" if boot else "n/a"
        print(f"  {variant:<18}{r['best_val_auc']:>9.3f}{auc_txt:>10}{ci_txt:>18}{lift:>12.2f}{r['verdict']:>9}")

    # ---- Interpretation (fixed rule, applied to the actual numbers) ----
    full_auc = all_results["full"]["test_skill"]["roc_auc"] or 0.0
    weather_auc = all_results["weather_only"]["test_skill"]["roc_auc"] or 0.0
    static_auc = all_results["static_only"]["test_skill"]["roc_auc"] or 0.0
    shuffled_auc = all_results["shuffled_weather"]["test_skill"]["roc_auc"] or 0.0
    any_pass = any(all_results[v]["verdict"] in ("PASS", "WEAK") for v in VARIANTS)

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  full={full_auc:.3f}  weather_only={weather_auc:.3f}  "
          f"static_only={static_auc:.3f}  shuffled_weather={shuffled_auc:.3f}")

    if static_auc >= full_auc - 0.01:
        print("  Soil/crop identity alone matches or beats the full model: static features (state/crop")
        print("  fingerprint), not weather, are doing the work in 'full'.")
    else:
        print("  Static features alone fall short of the full model: weather is contributing something.")

    if abs(shuffled_auc - full_auc) <= 0.02:
        print("  Shuffling train-year weather barely changes the result: 'full' is NOT relying on real")
        print("  year-to-year weather variation -- whatever signal it has comes from static features or")
        print("  the state/season mean level alone.")
    else:
        print("  Shuffling train-year weather changes the result meaningfully: real year-to-year weather")
        print("  variation is contributing to 'full', not just static features or the mean level.")

    print()
    if not any_pass:
        print("  FINAL READ: no variant passes or weakly passes the pre-registered bar. This is a genuine,")
        print("  thoroughly tested negative result -- raw input, class weighting, architecture, label")
        print("  drift, anomaly features with honest checkpointing, and now static-vs-weather ablation")
        print("  have each been tried and ruled out in turn. The RNN is not the bottleneck; the signal")
        print("  recoverable from this feature set (6 state-level weather variables + 3 soil + crop ID)")
        print("  at this spatial resolution does not support a reliable At-Risk classifier with this data.")
    else:
        passing = [v for v in VARIANTS if all_results[v]["verdict"] in ("PASS", "WEAK")]
        print(f"  FINAL READ: {passing} reach the pre-registered bar. Report this variant's numbers as the")
        print("  result, with the ablation table alongside it to show what the model is and isn't using.")

    out_path = output_dir(sequences_path) / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "input_kind": kind,
            "variants": {v: {k: r[k] for k in r if k != "epoch_log"} for v, r in all_results.items()},
            "epoch_logs": {v: all_results[v]["epoch_log"] for v in VARIANTS},
            "summary_aucs": {"full": full_auc, "weather_only": weather_auc,
                              "static_only": static_auc, "shuffled_weather": shuffled_auc},
            "any_variant_passes": any_pass,
        }, f, indent=2)
    print(f"\nResults written to: {out_path}")
    print("(On Colab this is /content -- download it before the runtime resets.)")


if __name__ == "__main__":
    main()