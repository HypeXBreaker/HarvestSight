# HarvestSight

**Can weekly weather and soil data flag crop-yield risk across Indian states? A hand-coded RNN says no. A simple moisture rule says "somewhat, in Kharif."**

HarvestSight is an attempt to predict seasonal crop-yield risk for Indian states from weekly weather and soil data, built to support UN Sustainable Development Goals 2 (Zero Hunger) and 13 (Climate Action). The project started with a recurrent neural network written from scratch. The network did not work. Instead of hiding that, this repository documents why it failed, what we tried next, and the modest but real signal we found with a transparent rule.

Everything is pure Python, standard library only. No NumPy, no ML frameworks.

---

## Contents

- [Key findings](#key-findings)
- [What the project set out to do](#what-the-project-set-out-to-do)
- [Data and problem setup](#data-and-problem-setup)
- [What we did, step by step](#what-we-did-step-by-step)
- [Why the neural network failed](#why-the-neural-network-failed)
- [What worked: a moisture rule](#what-worked-a-moisture-rule)
- [Results](#results)
- [Limitations](#limitations)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Interactive dashboard](#interactive-dashboard)
- [Future work](#future-work)
- [Data sources](#data-sources)
- [Author and license](#author-and-license)

---

## Key findings

All numbers below are ROC-AUC on years the models never saw (2017 to 2019) unless stated. An AUC of 0.5 is a coin flip, 1.0 is perfect.

| Model | Scope | Test AUC |
|---|---|---|
| Elman RNN, hand-coded | all seasons pooled | 0.529 |
| Elman RNN, hand-coded, 5 random starts | Kharif only | 0.526 (95% CI 0.472 to 0.579) |
| Moisture rule, no training | Kharif only | **0.595** (95% CI 0.515 to 0.667) |

- The neural network never beat its pre-defined success bar, in any configuration.
- A three-feature rule fixed in advance (how much wetter or drier than usual was the season?) did better, with nothing fitted to the labels.
- Cut into three tiers, the rule separates wet from dry Kharif seasons. In the test years, crops were **At-Risk 17.7% of the time in wet seasons and 34.9% in dry ones**, about twice as often.
- The pattern holds year by year: in 15 of 20 usable years the dry tier was riskier than the wet tier (sign test p = 0.021).
- The signal is season-specific and uneven. It is clearest in Kharif, weak in Rabi, and reverses in Summer. It works in some states and is flat in others.

The honest conclusion: at this data resolution, a simple, explainable rule beat a learned sequence model, and the effect is modest.

---

## What the project set out to do

The goal was an early-warning model. Given weather observed during a growing season, flag the crop-state combinations whose yield is likely to fall well below their usual trend, before harvest.

The design:

- A recurrent neural network (Elman RNN) reading 26 weekly weather steps per season.
- Static inputs per record: soil properties (pH, organic carbon, clay) and which crop it is.
- Everything hand-coded: forward pass, backpropagation through time, SGD, normalization, metrics. No ML libraries.
- Success judged by ROC-AUC and top-20% lift on held-out future years, not by accuracy.

Why it matters: yield shocks hit food security and farmer incomes first. If weather could reliably flag risky seasons, it would help with planning and early support.

---

## Data and problem setup

**Scope.** 28 Indian states, 1997 to 2019 (23 years), three seasons:

| Season | Window | Weeks |
|---|---|---|
| Kharif | Jun 1 to Nov 30 | 26 |
| Rabi | Oct 1 to Mar 31 | 26 |
| Summer | Mar 1 to Jun 30 | 17 |

**Inputs.**
- Weekly weather per state: rainfall, mean/max/min temperature, relative humidity, root-zone soil wetness.
- Soil per state: pH, organic carbon, clay content.
- A shortlist of 21 crops that appear in enough states to compare.
- One representative weather point per state, and state-level yields. This is a real limit on how much detail the data can carry.

**Dataset size.** 10,261 labeled (state, crop, season, year) records built from 1,377 distinct (state, season, year) weather observations. Every crop in the same state, season and year shares the same weather.

**Target.** A record is **At-Risk** when its yield falls clearly below that group's own trend. For each (state, crop, season), a linear trend of yield against year is fitted, and a record is At-Risk when its standardized residual is below -0.5. About 28% of records are At-Risk. Earlier three-class labels (Low, Moderate, High) could not be separated and were collapsed to two classes.

**Splits (by time, never shuffled).**

| Split | Years | Records |
|---|---|---|
| Train | 1997 to 2014 | 7,835 |
| Validation | 2015 to 2016 | 980 |
| Test | 2017 to 2019 | 1,446 |

---

## What we did, step by step

Each step below is a script in `scripts/`. Before each major run, the pass/fail rule was written into the script, so thresholds could not drift after seeing results.

1. **Data pipeline (01 to 06).** Fetch weather and soil, retry failed downloads, clean and aggregate yields, shortlist crops, derive trend-based risk labels, build weekly sequences.
2. **Train the RNN (07).** Three versions: raw weekly weather, then weekly anomalies, then validation-AUC checkpointing. All stayed near chance on test data.
3. **Diagnose (08).** Skip the network and test season-level weather anomalies directly against the labels. A small subset, mostly Kharif moisture anomalies, held up on held-out years. Temperature anomalies did not.
4. **Anomaly sequences (09).** Rebuild the RNN input as z-scores against each state's own climatology, so the network sees "wetter or drier than usual" instead of raw values.
5. **Ablation (10).** Four variants (full model, weather only, static features only, shuffled weather). None cleared the bar. Soil and crop identity alone scored 0.506, so there was no hidden shortcut either.
6. **Crop signal check (11).** A fixed moisture index with no training, evaluated by season and by crop. Includes a leakage-free test of whether focusing on fewer crops helps.
7. **Kharif-only RNN (12).** The fairest remaining test of the network: one season, five random starts, validation-AUC checkpoints. It failed too.
8. **Risk tiers and state map (13).** Turn the moisture index into three wet, medium and dry tiers and measure per-state sensitivity.
9. **Year-level robustness (14).** Treat the year as the unit of evidence, because states in the same year share monsoon conditions and are not independent.
10. **Dashboard (15).** An interactive, offline page that walks through the whole story.

Most scripts from 09 onward first re-derive earlier benchmarks (record counts, earlier AUCs) and print MATCH or DIFFERS before reporting anything new.

---

## Why the neural network failed

We tested the obvious explanations instead of guessing.

| Suspect | What we found |
|---|---|
| Mixing seasons confused the model | **Not the cause.** The Kharif-only RNN scored 0.526, the same as the pooled RNN (0.529). |
| It learned a soil or crop shortcut | **Not the cause.** A model given only soil and crop identity scored 0.506. |
| Too little data for the model size | **Likely, not proven.** About 4,200 Kharif training rows for a network reading 30 inputs a week with 16 hidden units. Training loss kept falling while validation AUC faded, the signature of overfitting. |
| Test years differ from training years | **Measured.** Test-year soil wetness ran about +0.65 standard deviations above the training climatology, humidity +0.34. Inputs at test time were partly outside what the network had seen. |
| Validation looked good, test did not | **Measured.** Checkpoints picked on a two-year validation set reached 0.61 to 0.69 AUC, then scored about 0.52 on test. Choosing the best of 20 epochs on a small set is optimistic. |

The training curves are reproduced in the dashboard's "Inside the model" tab so you can watch loss fall while validation AUC wanders.

A broader lesson: with state-level resolution, 23 years of data and shifting weather patterns, a model that must discover its own rules has too little to learn from. A rule built on prior knowledge (wetter than usual means lower risk) does better because it does not need to learn anything.

---

## What worked: a moisture rule

**The rule.** For each state and Kharif season, compare rainfall, root-zone wetness and humidity with that state's own training-year average (as z-scores). Average the three and flip the sign:

```
index = -mean(rainfall anomaly, soil wetness anomaly, humidity anomaly)
```

Higher means drier than usual, which means higher risk. Direction and weights were fixed in advance. Nothing is fitted to the labels, so there is nothing to memorize.

**Tiers.** Cut points at the one-third and two-thirds marks of the training-year index split seasons into low risk (wet), medium and high risk (dry). The cut points come from training years only.

---

## Results

### Moisture rule by season (AUC, 95% CI from a (state, year) cluster bootstrap)

| Season | Records | All years | Test years |
|---|---|---|---|
| Kharif | 5,445 | 0.608 [0.583, 0.640] | 0.595 |
| Rabi | 4,014 | 0.564 [0.532, 0.591] | 0.533 |
| Summer | 802 | 0.441 [0.395, 0.489] | 0.300 |

Summer reverses: wetter went with more risk. The same rule cannot be used across seasons.

### Does focusing on fewer crops help?

Crop-season groups were selected using only years up to 2016, then tested once on 2017 to 2019. The selected groups scored 0.603 against 0.595 for all of Kharif. The gain comes from choosing the season, not the crop, and the ranking among crops is within sampling noise. Rice (Kharif, 0.574) and Wheat (Rabi, 0.553, interval includes 0.5) are among the weaker groups.

### Kharif risk tiers (observed share of crop records At-Risk)

| Tier | Train | Validation | Test [95% CI] |
|---|---|---|---|
| Low risk (wet) | 22.5% | 11.6% | 17.7% [12.0, 22.5] |
| Medium | 26.8% | 30.4% | 28.9% [19.7, 38.1] |
| High risk (dry) | 36.7% | 50.3% | 34.9% [23.8, 45.9] |

The ordering is the same in all three splits. These are historical rates, not probabilities. The test years were wetter than the training years, so 60% of test rows fall in the wet tier.

### Does it hold year by year?

- Dry tier riskier than wet tier in **15 of 20 usable years** (one-sided sign test p = 0.021). Three years had too few records in one tier to compare.
- Pooled difference between dry and wet tiers: **+17.2 percentage points** (95% CI +10.6 to +23.3), from resampling whole years.
- Across all 23 years, national dryness and the national At-Risk share have a Spearman correlation of +0.47 (permutation p = 0.014).
- Five usable years ran the other way. This is a tendency, not a law.

### Where does it work?

For 24 states with enough Kharif data, 21 have an AUC above 0.5 and 8 are individually clear (Nagaland, Gujarat, Rajasthan, Mizoram, Maharashtra, Telangana, Karnataka, Chhattisgarh). It is flat in several others, including Punjab, Haryana, West Bengal, Bihar and Madhya Pradesh. Kerala runs the opposite way, but from few records.

The strong states are largely rain-dependent and the flat ones include heavily irrigated states. That fits the idea that irrigation weakens the link between weather and yield, but this project has no irrigation data, so it remains a hypothesis. The state results are exploratory: the top-ranked states are biased upward because they were chosen from many estimates.

The full tables (all states, all crop-season groups, every year) are in [`docs/HarvestSight_Final_Thesis.md`](docs/HarvestSight_Final_Thesis.md).

---

## Limitations

- **This is a historical pattern, not a forecast.** Tier rates are observed shares, not calibrated probabilities.
- **The feature choice is not a clean out-of-sample test.** The three moisture features were picked by a diagnostic that looked at all splits, and the 2017 to 2019 test years were examined in several rounds. The year-level check shows the pattern is not an accident of three years, but it does not remove this effect.
- **Coarse data.** One weather point per state and state-level yields hide local conditions, irrigation and microclimates.
- **Short record.** 23 years, and years are not fully independent (multi-year dry spells, trends).
- **Many analyses.** Decision rules were fixed per script, but the project ran many analyses. A one-sided p-value of 0.02 should be read as moderate evidence.
- **Season-specific.** The rule is for Kharif. Rabi is weak and Summer reverses.
- **Labels carry non-weather variation.** Trend-residual yield also reflects prices, inputs, pests and policy.
- **No irrigation, price or input data.**

---

## Repository structure

```
HarvestSight/
├── README.md
├── scripts/            numbered in run order, standard library only
├── cleaned_data/       intermediate JSON produced by the scripts
├── datasets/           raw downloaded data
├── model/              RNN weights written by script 07
├── results/            final result files copied from cleaned_data
├── dashboard/          harvestsight_dashboard.html
├── docs/
│   ├── HarvestSight_Final_Thesis.md
│   └── run_logs/       console output of each run
└── archive/            superseded versions
```

The scripts locate their data through the sibling `cleaned_data/` folder, so keep `scripts/` and `cleaned_data/` side by side.

---

## How to run

Requirements: Python 3.8 or newer. No packages to install.

Run the scripts in order from the repository root. Scripts 01 to 06 download and prepare data. Later scripts print a self-check against earlier results before reporting their own findings.

| # | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 01 | `01_fetch_datasets.py` | Download weather and soil | online sources | `datasets/` |
| 01b | `01b_retry_failed_soil.py` | Retry failed soil downloads | `datasets/` | `datasets/` |
| 02 | `02_retry_soil_weather.py` | Retry remaining soil/weather gaps | `datasets/` | `datasets/` |
| 03 | `03_clean_and_aggregate.py` | Clean and aggregate yields | `datasets/` | `state_crop_season_year_yield.json` |
| 04 | `04_crop_shortlist.py` | Choose the 21-crop shortlist | yield file | `crop_shortlist.json` |
| 05 | `05_derive_risk_labels.py` | Trend-residual risk labels | yield file, shortlist | `yield_with_risk_labels.json` |
| 06 | `06_build_weekly_sequences.py` | Weekly weather sequences | labels, weather, soil | `rnn_sequences.json` |
| 07 | `07_train_model.py` | Train and evaluate the RNN | sequences, shortlist | `model/harvestsight_rnn*.json` |
| 08 | `08_diagnostic.py` | Season-level anomaly diagnostic | `rnn_sequences.json` | `signal_diagnostic_results.json` |
| 09 | `09_build_anomaly_sequences.py` | Weekly z-score sequences | `rnn_sequences.json` | `rnn_sequences_anomaly.json`, `weather_climatology.json` |
| 10 | `10_ablation_study.py` | RNN ablation variants | anomaly sequences | console output |
| 11 | `11_crop_signal_check.py` | Moisture index by season and crop | `rnn_sequences.json` | `crop_signal_results.json` |
| 12 | `12_kharif_rnn.py` | Kharif-only RNN, 5 seeds | anomaly sequences, shortlist | `kharif_rnn_results.json` |
| 13 | `13_kharif_suitability.py` | Risk tiers and per-state map | `rnn_sequences.json` | `kharif_suitability.json` |
| 14 | `14_year_robustness.py` | Year-level robustness | `rnn_sequences.json` | `kharif_year_robustness.json` |
| 15 | `15_build_dashboard.py` | Build the dashboard | files from 13 and 14 | `harvestsight_dashboard.html` |

Runtimes: scripts 08, 11, 13, 14 and 15 finish in seconds. Script 12 takes about 7 minutes per random start (about 35 minutes for five) on a laptop CPU. On Colab, upload the needed files to `/content` and run a script as a single cell; the scripts also search the current folder and `/content`.

---

## Interactive dashboard

`python scripts/15_build_dashboard.py` writes a single self-contained `harvestsight_dashboard.html` with all data embedded. Open it in any browser, no server required.

| Tab | What it shows |
|---|---|
| Tour | A six-step walkthrough of the whole project |
| Inside the model | The rule step by step, a replay of RNN training epoch by epoch, and the tested reasons the network failed |
| Tiers | At-Risk rate by tier with intervals, plus a tool to see which tier an index value falls in |
| Map | Every state and year colored by wet, medium or dry season |
| States | Where the rule tracks risk, with each state's history |
| Years | Dry-versus-wet comparison for every year |

---

## Future work

- District-level yields and several weather points per state to restore spatial detail.
- Irrigation share and input data, to test whether irrigation explains the flat states.
- Year-blocked cross-validation across all 23 years instead of a fixed three-year test.
- Separate rules for Rabi and Summer, and adjustment for the shifting climate.
- Crop-specific models once there is enough data per crop. Rice and Wheat had about 460 and 500 records, too few for a stable held-out estimate.

---

## Data sources

- **Weather:** NASA POWER, https://power.larc.nasa.gov/
- **Soil:** SoilGrids, https://soilgrids.org/
- **Crop yields:** [add the original source and link here]

Check each source's terms of use and citation requirements before redistributing data.

---

## Author and license

**Author:** Aditya Mohapatra & Team

**License:** [Add a LICENSE file and name it here]