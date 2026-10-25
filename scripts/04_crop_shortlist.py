"""
HarvestSight — Phase 3 / Script 04
Crop shortlist (national crop-viability filter).

Phase 2's output has 77 distinct crops, but most are hyper-regional -
grown seriously in only a handful of states. This script ranks every
crop by how many distinct states it appears in at all (any season, any
single year - simple presence, no minimum-years sub-requirement, per
the confirmed method), then keeps only crops covering >= MIN_STATES
states as viable targets for a pan-India model.

Input:  cleaned_data/state_crop_season_year_yield.json  (Phase 2 output)
Output: cleaned_data/crop_viability_report.json  - ALL 77 crops, ranked,
        with their state-coverage counts, so the cutoff point is visible
        and checkable rather than a black-box filter.
        cleaned_data/crop_shortlist.json  - just the crops clearing the
        threshold, alphabetically sorted (deterministic ordering for
        later one-hot encoding).

Stdlib only (json, os).
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEANED_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")

INPUT_PATH = os.path.join(CLEANED_DIR, "state_crop_season_year_yield.json")
REPORT_PATH = os.path.join(CLEANED_DIR, "crop_viability_report.json")
SHORTLIST_PATH = os.path.join(CLEANED_DIR, "crop_shortlist.json")

# Locked from the original build - the "strict" tier, chosen over looser
# balanced/broad alternatives.
MIN_STATES = 19


def main():
    print("=" * 70)
    print("HarvestSight Phase 3 / Script 04 - crop shortlist")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Phase 2 output not found at {INPUT_PATH}")
        print("        Run 03_clean_and_aggregate_yield.py first.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records):,} state/crop/season/year cells from Phase 2 output.")

    # crop -> set of states, set of years, count of cells
    crop_states = {}
    crop_years = {}
    crop_cell_count = {}

    for rec in records:
        crop = rec["crop"]
        state = rec["state"]
        year = rec["year"]

        crop_states.setdefault(crop, set()).add(state)
        crop_years.setdefault(crop, set()).add(year)
        crop_cell_count[crop] = crop_cell_count.get(crop, 0) + 1

    # Build the full ranked report - every crop, sorted by state coverage
    # descending, crop name ascending as a stable tiebreak.
    report = []
    for crop, states in crop_states.items():
        report.append({
            "crop": crop,
            "n_states": len(states),
            "n_cells": crop_cell_count[crop],
            "n_years": len(crop_years[crop]),
            "included": len(states) >= MIN_STATES,
            "states": sorted(states),
        })

    report.sort(key=lambda r: (-r["n_states"], r["crop"]))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Shortlist - just the crops clearing the threshold, alphabetically
    # sorted for a deterministic one-hot ordering downstream.
    shortlist_crops = sorted(r["crop"] for r in report if r["included"])

    shortlist_output = {
        "threshold_min_states": MIN_STATES,
        "n_crops": len(shortlist_crops),
        "crops": shortlist_crops,
    }
    with open(SHORTLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(shortlist_output, f, indent=2)

    # --- Console report: the FULL ranked list, not just the winners ---
    print(f"\n{'crop':45s} {'n_states':>8s}  {'n_cells':>7s}  {'n_years':>7s}  included")
    print("-" * 80)
    printed_cutoff_marker = False
    for r in report:
        if not r["included"] and not printed_cutoff_marker:
            print(f"{'--- cutoff: >= ' + str(MIN_STATES) + ' states required ---':80s}")
            printed_cutoff_marker = True
        marker = "YES" if r["included"] else "no"
        print(f"{r['crop']:45s} {r['n_states']:>8d}  {r['n_cells']:>7d}  {r['n_years']:>7d}  {marker}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total distinct crops:     {len(report)}")
    print(f"Threshold:                >= {MIN_STATES} states covered")
    print(f"Crops in shortlist:       {len(shortlist_crops)}")
    print(f"   (benchmark from the prior validated run: 21 crops)")
    print(f"\nFull ranked report written to {REPORT_PATH}")
    print(f"Shortlist written to {SHORTLIST_PATH}")


if __name__ == "__main__":
    main()