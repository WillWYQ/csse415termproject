"""
run_all_configs.py
==================
Batch orchestrator for the "Will You Get a Job?" project.

Runs a sweep of binary-target configurations by calling run() from
train_and_export.py for each experiment definition below.

Each experiment varies three things:
  1. target_col        — which binary column we are predicting
  2. feature_group     — which "optional" feature columns are retained
  3. drop_cols         — derived automatically from the above two

Experiment Groups
-----------------
  Group A  "restricted_*"
      Drop all job-search-behavior columns
      (Networking_Events_Attended, Primary_Search_Platform,
       Months_Searching, Applications_Submitted).
      Predict purely from student background.

  Group B  "full_*"
      Keep all job-search-behavior columns.
      Predict from background + search effort/platform.

  Group C  "cascade_*"  ← bonus meaningful combinations
      Cascade prediction: earlier-stage outcome columns are
      kept as features for a later-stage target (not leakage —
      the earlier stage happened before the later one).
        cascade_2nd  : First_Round_Interviews → feature for target Second_Round
        cascade_offer: First_Round + Second_Round  → features for target Offer_Received
        cascade_offer_restricted: same but without behavior cols

Running
-------
    python run_all_configs.py                       # all experiments
    python run_all_configs.py --data path.csv       # override data path
    python run_all_configs.py --quick               # tiny grids (dev mode)
    python run_all_configs.py --no-fe               # skip polynomial FE
    python run_all_configs.py --cv 3                # override CV folds
    python run_all_configs.py --only A              # run only group A
    python run_all_configs.py --only B C cascade    # run groups B, C
    python run_all_configs.py --list                # print configs and exit

Output
------
Each experiment writes its own output directory (auto-named by
train_and_export._resolve_output_dir), so nothing is ever overwritten.
A run-level summary table is printed at the end to stdout.

AI Usage
--------
This file was developed with the assistance of generative AI based on our
previous work. We ensured we understand and verify all AI-generated code,
and we are responsible for its correctness and suitability for our project.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── Locate train_and_export.py ───────────────────────────────────────────────
# Supports running from any working directory as long as train_and_export.py
# is in the same folder as this script.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import train_and_export as _tae  # noqa: E402

# Snapshot of the default settings — never mutated after this point.
# We deepcopy it per experiment; _tae.CONFIG is patched in-place so that
# any module (e.g. prepare.py) that holds a reference to the same dict
# object picks up the correct per-experiment settings.
_BASE_CONFIG = copy.deepcopy(_tae.CONFIG)
run = _tae.run


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 1 — COLUMN SETS
# ══════════════════════════════════════════════════════════════════════════════

# Columns that are ALWAYS dropped regardless of experiment:
#   - Student_ID         : pure row identifier, zero predictive signal
#   - Time_to_Offer_Days : 60 k+ NaN; also a post-outcome leaker
#   - Offer_Salary       : known only after offer → leakage; 60%+ missing
#   - Company_Size_Offered: same
#   - Role_Relevance     : same
#   - Accepted_Offer     : downstream of Offer_Received → leakage
_BASE_DROP = [
    "Student_ID",
    "Time_to_Offer_Days",
    "Offer_Salary",
    "Company_Size_Offered",
    "Role_Relevance",
    "Accepted_Offer",
]

# All three raw outcome columns.  For each experiment we drop the ones
# that are NOT the current target (to prevent inter-outcome leakage).
_ALL_OUTCOMES = [
    "First_Round_Interviews",
    "Second_Round_Interviews",
    "Offer_Received",
]

# Job-search behaviour columns.  Dropped in "restricted" experiments,
# kept in "full" and "cascade" experiments.
_BEHAVIOR_COLS = [
    "Networking_Events_Attended",
    "Primary_Search_Platform",
    "Months_Searching",
    "Applications_Submitted",
]


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 2 — EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Each experiment is a dict with:
#   name        : short unique label (used in summary table)
#   group       : letter used by --only filter  (A / B / C)
#   description : one-line human description
#   target_col  : binary target column
#   extra_drop  : additional columns to drop ON TOP of _BASE_DROP
#                 (do NOT include the target_col itself here — prepare()
#                  extracts it as the target before any dropping)
#
# NOTE: _BASE_DROP is always applied; extra_drop is merged on top.
# The target_col must NOT appear in either list.
# ─────────────────────────────────────────────────────────────────────────────

def _other_outcomes(target: str) -> List[str]:
    """Return the two outcome cols that are NOT the current target."""
    return [c for c in _ALL_OUTCOMES if c != target]


EXPERIMENTS: List[Dict[str, Any]] = [

    # ── Group A: restricted (drop behavior cols) ──────────────────────────────
    {
        "name":        "A1_restricted_first_round",
        "group":       "A",
        "description": "Target=First_Round_Interviews | background-only features (no behavior cols)",
        "target_col":  "First_Round_Interviews",
        "extra_drop":  _other_outcomes("First_Round_Interviews") + _BEHAVIOR_COLS,
    },
    {
        "name":        "A2_restricted_second_round",
        "group":       "A",
        "description": "Target=Second_Round_Interviews | background-only features (no behavior cols)",
        "target_col":  "Second_Round_Interviews",
        "extra_drop":  _other_outcomes("Second_Round_Interviews") + _BEHAVIOR_COLS,
    },
    {
        "name":        "A3_restricted_offer",
        "group":       "A",
        "description": "Target=Offer_Received | background-only features (no behavior cols)",
        "target_col":  "Offer_Received",
        "extra_drop":  _other_outcomes("Offer_Received") + _BEHAVIOR_COLS,
    },

    # ── Group B: full (keep behavior cols) ────────────────────────────────────
    {
        "name":        "B1_full_first_round",
        "group":       "B",
        "description": "Target=First_Round_Interviews | background + behavior features",
        "target_col":  "First_Round_Interviews",
        "extra_drop":  _other_outcomes("First_Round_Interviews"),
    },
    {
        "name":        "B2_full_second_round",
        "group":       "B",
        "description": "Target=Second_Round_Interviews | background + behavior features",
        "target_col":  "Second_Round_Interviews",
        "extra_drop":  _other_outcomes("Second_Round_Interviews"),
    },
    {
        "name":        "B3_full_offer",
        "group":       "B",
        "description": "Target=Offer_Received | background + behavior features",
        "target_col":  "Offer_Received",
        "extra_drop":  _other_outcomes("Offer_Received"),
    },

    # ── Group C: cascade predictions (earlier stages as features) ─────────────
    #
    # Rationale: knowing that a student got a first-round interview is a
    # legitimate (non-leaking) feature for predicting second-round or offer,
    # because the first round happens *before* the later outcomes.
    {
        "name":        "C1_cascade_2nd_with_behavior",
        "group":       "C",
        "description": (
            "Target=Second_Round_Interviews | "
            "First_Round_Interviews kept as feature (cascade) + behavior cols"
        ),
        "target_col":  "Second_Round_Interviews",
        # Only drop Offer_Received (post-target leaker); keep First_Round as feature
        "extra_drop":  ["Offer_Received"],
    },
    {
        "name":        "C2_cascade_2nd_restricted",
        "group":       "C",
        "description": (
            "Target=Second_Round_Interviews | "
            "First_Round_Interviews kept as feature (cascade), no behavior cols"
        ),
        "target_col":  "Second_Round_Interviews",
        "extra_drop":  ["Offer_Received"] + _BEHAVIOR_COLS,
    },
    {
        "name":        "C3_cascade_offer_with_behavior",
        "group":       "C",
        "description": (
            "Target=Offer_Received | "
            "First_Round + Second_Round kept as features (cascade) + behavior cols"
        ),
        "target_col":  "Offer_Received",
        # Keep both interview-round cols as features; nothing else to drop
        "extra_drop":  [],
    },
    {
        "name":        "C4_cascade_offer_restricted",
        "group":       "C",
        "description": (
            "Target=Offer_Received | "
            "First_Round + Second_Round kept as features (cascade), no behavior cols"
        ),
        "target_col":  "Offer_Received",
        "extra_drop":  _BEHAVIOR_COLS,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 3 — CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_config(
    experiment: Dict[str, Any],
    base_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge the base CONFIG with per-experiment settings and any CLI overrides.

    Drop-list construction
    ----------------------
    final_drop = _BASE_DROP + extra_drop   (target_col forcibly excluded)

    The target_col is explicitly removed from the drop list even if the user
    accidentally listed it, because prepare() needs the column to exist in
    order to extract it as the target.
    """
    cfg = copy.deepcopy(_BASE_CONFIG)

    # Apply any CLI-level overrides first
    cfg.update(base_overrides)

    # Apply experiment-specific settings
    cfg["target_mode"] = "binary"
    cfg["target_col"]  = experiment["target_col"]

    # Build the final drop list, guarding against accidentally dropping target
    target = experiment["target_col"]
    raw_drop = list(dict.fromkeys(_BASE_DROP + experiment["extra_drop"]))  # dedupe, order-stable
    cfg["drop_cols"] = [c for c in raw_drop if c != target]

    # Give the output directory a descriptive name based on experiment name
    cfg["output_dir_base"] = f"exported_models__{experiment['name']}"

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 4 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a colour-free ASCII summary table of all experiment outcomes."""
    col_widths = {
        "name":   max(len(r["name"])   for r in results),
        "status": max(len(r["status"]) for r in results),
        "time":   8,
        "dir":    max(len(r["out_dir"]) for r in results),
    }
    col_widths["name"]   = max(col_widths["name"],   4)
    col_widths["status"] = max(col_widths["status"], 6)

    sep  = (
        f"+-{'-' * col_widths['name']}-+-{'-' * col_widths['status']}-"
        f"+-{'-' * col_widths['time']}-+-{'-' * col_widths['dir']}-+"
    )
    header = (
        f"| {'Name':<{col_widths['name']}} | {'Status':<{col_widths['status']}} "
        f"| {'Time(m)':>{col_widths['time']}} | {'Output Dir':<{col_widths['dir']}} |"
    )

    print("\n" + "=" * 70)
    print("  BATCH SUMMARY")
    print("=" * 70)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        t = f"{r['elapsed_min']:.1f}" if r["elapsed_min"] is not None else "—"
        print(
            f"| {r['name']:<{col_widths['name']}} "
            f"| {r['status']:<{col_widths['status']}} "
            f"| {t:>{col_widths['time']}} "
            f"| {r['out_dir']:<{col_widths['dir']}} |"
        )
    print(sep)

    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = len(results) - n_ok
    print(f"\n  {n_ok}/{len(results)} experiments succeeded, {n_fail} failed.")
    if n_fail:
        print("\n  Failed experiments:")
        for r in results:
            if r["status"] != "OK":
                print(f"    [{r['name']}]  {r.get('error', 'unknown error')}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 5 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all binary-target experiments for 'Will You Get a Job?'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default=None, metavar="PATH",
        help="Path to raw CSV.  Overrides CONFIG['data_path'].",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Tiny coarse grids for fast dev iteration (~1 min per experiment).",
    )
    p.add_argument(
        "--no-fe", action="store_true",
        help="Skip polynomial feature engineering (faster; weaker linear models).",
    )
    p.add_argument(
        "--cv", type=int, default=None, metavar="N",
        help="Number of cross-validation folds.",
    )
    p.add_argument(
        "--only", nargs="+", metavar="GROUP",
        help=(
            "Run only the specified group(s). "
            "Choices: A  B  C  (or any subset, space-separated). "
            "E.g.  --only A B"
        ),
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all experiments and exit without running anything.",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        print(f"\n{'Experiments defined in run_all_configs.py':^70}")
        print("=" * 70)
        for exp in EXPERIMENTS:
            drop_preview = ", ".join(
                [c for c in _BASE_DROP + exp["extra_drop"] if c != exp["target_col"]]
            )
            print(f"\n  [{exp['name']}]  (Group {exp['group']})")
            print(f"  {exp['description']}")
            print(f"  target_col : {exp['target_col']}")
            print(f"  drop_cols  : {drop_preview}")
        print()
        sys.exit(0)

    # ── Filter experiments by group ───────────────────────────────────────────
    selected = EXPERIMENTS
    if args.only:
        allowed = {g.upper() for g in args.only}
        selected = [e for e in EXPERIMENTS if e["group"].upper() in allowed]
        if not selected:
            print(
                f"ERROR: --only {args.only!r} matched no experiments. "
                f"Valid groups: A B C"
            )
            sys.exit(1)

    # ── Build base CLI overrides (applied to every experiment) ────────────────
    base_overrides: Dict[str, Any] = {}
    if args.data  is not None: base_overrides["data_path"]    = args.data
    if args.cv    is not None: base_overrides["cv"]           = args.cv
    if args.quick:             base_overrides["_quick_mode"]  = True
    if args.no_fe:             base_overrides["do_fe"]        = False

    # ── Banner ────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("=" * 70)
    print("  Will You Get a Job? — Batch Config Runner")
    print(f"  Started : {ts}")
    print(f"  Experiments to run : {len(selected)}")
    print(f"  Quick mode  : {base_overrides.get('_quick_mode', False)}")
    print(f"  FE enabled  : {not args.no_fe}")
    print(f"  CV folds    : {base_overrides.get('cv', _BASE_CONFIG['cv'])}")
    print("=" * 70)

    # ── Run loop ──────────────────────────────────────────────────────────────
    summary: List[Dict[str, Any]] = []

    for idx, exp in enumerate(selected, start=1):
        cfg = _build_config(exp, base_overrides)
        out_dir = f"{cfg['output_dir_base']}__{cfg['target_col']}"

        print(f"\n{'─' * 70}")
        print(f"  Experiment {idx}/{len(selected)}:  [{exp['name']}]")
        print(f"  {exp['description']}")
        print(f"  target_col : {cfg['target_col']}")
        print(f"  drop_cols  : {cfg['drop_cols']}")
        print(f"  output_dir : {out_dir}")
        print(f"{'─' * 70}\n")

        t0      = time.perf_counter()
        status  = "OK"
        err_msg = ""

        try:
            # ── Patch the module-level CONFIG in-place ────────────────────────
            # prepare.py (and any other sibling module) may hold a reference to
            # the same dict object as train_and_export.CONFIG.  Updating it
            # in-place ensures those references see the correct per-experiment
            # settings (target_col, drop_cols, etc.) even if they imported the
            # dict at module load time.
            _tae.CONFIG.clear()
            _tae.CONFIG.update(cfg)

            run(cfg)
        except Exception:
            status  = "FAILED"
            err_msg = traceback.format_exc().strip().splitlines()[-1]
            print(f"\n[batch] ERROR in experiment [{exp['name']}]:\n")
            traceback.print_exc()

        elapsed_min = (time.perf_counter() - t0) / 60.0

        summary.append({
            "name":        exp["name"],
            "status":      status,
            "elapsed_min": elapsed_min,
            "out_dir":     out_dir,
            "error":       err_msg,
        })

        print(
            f"\n[batch] Experiment [{exp['name']}] finished "
            f"— status={status}, time={elapsed_min:.1f} min"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(summary)


if __name__ == "__main__":
    main()