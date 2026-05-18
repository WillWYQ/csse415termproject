"""
train_and_export.py
===================
Master orchestrator for the "Will You Get a Job?" project.

Responsibilities
----------------
1. CONFIG      – single source of truth for every knob in the pipeline:
                   data path, columns to drop, target mode, prepare settings,
                   feature-engineering settings, training/CV settings.
2. LOAD        – read the raw CSV and hand it to prepare().
3. PREPARE     – call prepare(df, config) → PreparedData.
4. SPECS       – start from build_model_specs(), then apply any project-specific
                 overrides (e.g. inject class_weight, narrow a grid).
5. TRAIN       – call train_and_compare(data, specs, config) → CompareResult.
6. EXPORT      – serialise every fitted model, write comparison.csv,
                 write a README.md summarising the run, and save the full
                 console log to run.log — all under an auto-named output dir.

Output directory naming
-----------------------
The output directory is derived automatically from the target configuration
so that different runs never collide and their purpose is self-documenting:

    exported_models__4class/                   ← target_mode='4class'
    exported_models__binary__Offer_Received/   ← target_mode='binary'

Within that directory every run appends to (or creates) run.log so you keep
a permanent record of every training session without overwriting older logs.

Running
-------
    python train_and_export.py                  # uses default DATA_PATH
    python train_and_export.py --data path.csv  # override data path
    python train_and_export.py --quick          # small fast grids (dev mode)
    python train_and_export.py --no-fe          # skip polynomial FE
    python train_and_export.py --target binary  # binary Offer_Received target
    python train_and_export.py --cv 3           # override cross-validation folds

File layout expected
--------------------
    train_and_export.py   <- this file
    prepare.py            <- data-preparation module
    train_and_compare.py  <- model-training / comparison module
    <DATA_PATH>           <- raw CSV

AI Usage
--------
This file was developed with the assistance of generative AI based on our
previous work. We ensured we understand and verify all AI-generated code,
and we are responsible for its correctness and suitability for our project.
"""

from __future__ import annotations

import os

# ── Thread-count control (set before numpy / sklearn are imported) ────────────
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

import argparse
import io
import pathlib
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os
import math
import joblib
import numpy as np
import pandas as pd

# 物理核心数（不含超线程），留 2 个给系统
n_physical = joblib.cpu_count(only_physical_cores=True)
N_JOBS = max(1, n_physical - 2)
print(f"[config] physical cores={n_physical}, using n_jobs={N_JOBS}")

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from prepare import prepare, PreparedData
from train_and_compare import (
    CompareResult,
    ModelSpec,
    build_model_specs,
    train_and_compare,
    zoom_int,
    zoom_log,
)


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 1 — MASTER CONFIG
# ══════════════════════════════════════════════════════════════════════════════
#
# Every setting that controls the pipeline lives here.
# Neither prepare.py nor train_and_compare.py hard-codes any of these values.
#
# Key conventions
# ---------------
#   drop_cols      : columns removed BEFORE the target is derived.
#                    Non-existent names are silently ignored so you can keep
#                    this list stable across dataset versions.
#
#   target_mode    : '4class' — synthesise application_result {0,1,2,3} from
#                               First_Round_Interviews, Second_Round_Interviews,
#                               Offer_Received.
#                   'binary'  — use the column named in target_col directly.
#
#   target_col     : only used when target_mode == 'binary'.
#
#   output_dir_base : base prefix; the actual output dir is derived from
#                     target_mode and target_col automatically.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {

    # ── I/O ──────────────────────────────────────────────────────────────────
    "data_path":       "../data/job_search_platform_efficacy_100k.csv",
    "output_dir_base": "exported_models",   # suffix added automatically

    # ── Target ───────────────────────────────────────────────────────────────
    "target_mode": "binary",                # '4class' or 'binary
    "target_col":  "Offer_Received",        # only used when target_mode='binary'

    # ── Columns to DROP before any processing ────────────────────────────────
    #
    #   Student_ID              — identifier; carries zero signal
    #   Time_to_Offer_Days      — 60 k+ NaN; post-outcome leaker
    #   Offer_Salary            — known only after offer received -> leakage;
    #                             60 %+ missing
    #   Company_Size_Offered    — same
    #   Role_Relevance          — same
    #   Accepted_Offer          — downstream of Offer_Received -> leakage
    #
    #   NOTE: First_Round_Interviews, Second_Round_Interviews, Offer_Received
    #   are NOT listed here because prepare() uses them to build the 4-class
    #   target and removes them automatically after that.
    #
    "drop_cols": [
        "",
        "Student_ID",
        "Time_to_Offer_Days",
        "Offer_Salary",
        "Company_Size_Offered",
        "Role_Relevance",
        "Accepted_Offer",
    ],

    # ── Prepare settings ─────────────────────────────────────────────────────
    "test_size":    0.20,
    "random_state": 0,

    # ── Feature engineering ───────────────────────────────────────────────────
    "do_fe":               True,
    "fe_degree":           2,
    "fe_interaction_only": True,     # cross-terms only; keeps pool manageable
    "fe_kind":             "logreg", # 'logreg' or 'knn'
    "fe_cv_folds":         3,
    "fe_lr_max_iter":      500,

    # ── Training / cross-validation ───────────────────────────────────────────
    "cv":      5,
    "scoring": "accuracy",

    # ── Analysis output ───────────────────────────────────────────────────────
    "n_examples": 3,
    "top_n_fi":   12,

    # ── Quick mode (--quick flag) ─────────────────────────────────────────────
    "_quick_mode": False,
}


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 2 — OUTPUT DIRECTORY NAMING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_output_dir(config: Dict[str, Any]) -> pathlib.Path:
    """
    Derive the output directory path from the target configuration.

    Examples
    --------
    target_mode='4class'
        -> exported_models__4class/
    target_mode='binary', target_col='Offer_Received'
        -> exported_models__binary__Offer_Received/
    """
    base = config["output_dir_base"]
    mode = config["target_mode"]

    if mode == "4class":
        suffix = "4class"
    else:
        col    = config.get("target_col", "target")
        suffix = f"binary__{col}"

    return pathlib.Path(f"{base}__{suffix}")


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 3 — TEE LOGGER  (console + file simultaneously)
# ══════════════════════════════════════════════════════════════════════════════

class _Tee(io.TextIOBase):
    """
    Forwards every write to both the original stream and a log file.

    The log file is opened in **append** mode so multiple runs accumulate in
    the same file, each separated by a timestamped header written at install
    time.

    Usage (internal only)
    ---------------------
        tee_out, tee_err = _install_tee(log_path)
        ...
        _remove_tee(tee_out, tee_err)   # restores sys.stdout / sys.stderr
    """

    def __init__(self, original: Any, log_path: pathlib.Path) -> None:
        self._orig = original
        self._file = log_path.open("a", encoding="utf-8")

    def write(self, text: str) -> int:
        self._orig.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._orig.flush()
        self._file.flush()

    def close(self) -> Any:
        """Close the log file and return the original stream."""
        self._file.close()
        return self._orig


def _install_tee(log_path: pathlib.Path):
    """
    Replace sys.stdout and sys.stderr with Tee objects that mirror output to
    log_path.  A timestamped header is written first so runs are easy to
    distinguish inside the file.
    """
    # Write session header before the Tee takes over
    with log_path.open("a", encoding="utf-8") as f:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write(f"\n{'=' * 70}\n  Run started  {ts}\n{'=' * 70}\n")

    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = tee_out
    sys.stderr = tee_err
    return tee_out, tee_err


def _remove_tee(tee_out: _Tee, tee_err: _Tee) -> None:
    """Restore sys.stdout and sys.stderr and close the log file handles."""
    sys.stdout = tee_out.close()
    sys.stderr = tee_err.close()


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 4 — MODEL SPEC DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _kernel_svm_pipeline():
    """Build the Nystroem + LogisticRegression pipeline used by Kernel SVM."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("nystroem", Nystroem(kernel="rbf", random_state=0)),
        ("clf",      LogisticRegression(
            max_iter=20000, random_state=0, class_weight="balanced"
        )),
    ])


def _build_project_specs(config: Dict[str, Any]) -> List[ModelSpec]:
    """
    Return the full list of ModelSpec objects for this project.

    Starts from the canonical defaults in train_and_compare.build_model_specs(),
    then layers on project-specific overrides:
      * class_weight='balanced' injected into every estimator that supports it,
        to address the consistent under-performance on class 1 (first-round only).
      * Quick mode shrinks every coarse grid to 2-4 combos for fast iteration.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    quick = config.get("_quick_mode", False)

    # ── refine functions ─────────────────────────────────────────────────────

    def _no_round2(_):
        return {}

    def _refine_knn(best):
        k = int(best["n_neighbors"])
        return {
            "n_neighbors": zoom_int(k, step=1, n=3),
            "weights":     [best["weights"]],
            "metric":      [best["metric"]],
        }

    def _refine_ridge(best):
        return {"C": zoom_log(best["C"], n=5, radius_decades=0.5)}

    def _refine_lasso(best):
        return {"C": zoom_log(best["C"], n=5, radius_decades=0.5)}

    def _refine_svm(best):
        return {
            "estimator__C":    zoom_log(best["estimator__C"], n=5, radius_decades=0.5),
            "estimator__loss": [best["estimator__loss"]],
        }

    def _refine_dt(best):
        d = best["max_depth"]
        return {
            "max_depth":         (zoom_int(d, step=1, n=2) if d is not None
                                  else [6, 8, 10, 12, None]),
            "min_samples_split": [best["min_samples_split"]],
            "min_samples_leaf":  [best["min_samples_leaf"]],
        }

    def _refine_rf(best):
        return {
            "n_estimators": zoom_int(int(best["n_estimators"]), step=1,  n=10),
            "max_depth":    zoom_int(int(best["max_depth"]),    step=1,  n=2),
        }

    def _refine_gbt(best):
        return {
            "learning_rate": [float(best["learning_rate"])],
            "n_estimators":  zoom_int(int(best["n_estimators"]), step=10, n=3),
            "max_depth":     zoom_int(int(best["max_depth"]),    step=1,  n=2),
        }

    def _refine_kernel_svm(best):
        return {
            "nystroem__gamma":        zoom_log(best["nystroem__gamma"],
                                               n=4, radius_decades=0.75),
            "clf__C":                 zoom_log(best["clf__C"],
                                               n=4, radius_decades=0.75),
            "nystroem__n_components": [best["nystroem__n_components"]],
        }

    # ── specs ────────────────────────────────────────────────────────────────

    return [

        ModelSpec(
            name      = "KNN (tuned)",
            estimator = KNeighborsClassifier(n_jobs=N_JOBS),
            dataset   = "FE",
            param_grid_1 = {
                "n_neighbors": (np.array([3, 11, 21]) if quick
                                else np.arange(3, 32, 2)),
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan"],
            },
            refine_fn = _refine_knn,
        ),

        ModelSpec(
            name      = "Logistic Regression",
            estimator = LogisticRegression(
                max_iter=10000, random_state=0, class_weight="balanced"
            ),
            dataset   = "FE",
            param_grid_1 = {},
            refine_fn = _no_round2,
        ),

        ModelSpec(
            name      = "Ridge Logistic Regression",
            estimator = LogisticRegression(
                max_iter=10000, random_state=0,
                class_weight="balanced",
            ),
            dataset   = "FE",
            param_grid_1 = {
                "C": (np.array([0.01, 1.0, 100.0]) if quick else
                      1 / np.array([1e-4, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5])),
            },
            refine_fn = _refine_ridge,
        ),

        ModelSpec(
            name      = "Lasso Logistic Regression",
            estimator = LogisticRegression(
                l1_ratio=1, solver='saga', max_iter=10000, random_state=0,
                class_weight="balanced",
            ),
            dataset   = "FE",
            param_grid_1 = {
                "C": (np.array([0.01, 1.0, 100.0]) if quick else
                      1 / np.array([1e-4, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5])),
            },
            refine_fn = _refine_lasso,
        ),

        ModelSpec(
            name      = "LinearSVC",
            estimator = CalibratedClassifierCV(
                LinearSVC(max_iter=50000, tol=1e-3, random_state=0, class_weight="balanced")
            ),
            dataset   = "FE",
            param_grid_1 = {
                "estimator__C": (
                    [0.01, 1.0, 100.0] if quick else
                    [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, 50, 100, 1000]
                ),
                "estimator__loss": ["hinge", "squared_hinge"],
            },
            refine_fn = _refine_svm,
        ),

        ModelSpec(
            name      = "Decision Tree (tuned)",
            estimator = DecisionTreeClassifier(
                random_state=0, class_weight="balanced"
            ),
            dataset   = "base",
            param_grid_1 = {
                "max_depth":         ([3, 8, None] if quick else
                                      [2, 3, 4, 5, 6, 8, 10, 15, None]),
                "min_samples_split": [2, 10, 50],
                "min_samples_leaf":  [1, 5, 20],
            },
            refine_fn = _refine_dt,
        ),

        ModelSpec(
            name      = "Random Forest (tuned)",
            estimator = RandomForestClassifier(
                random_state=0, n_jobs=N_JOBS, class_weight="balanced"
            ),
            dataset   = "base",
            param_grid_1 = {
                "n_estimators": (np.array([10, 100]) if quick 
                     else (np.arange(1, 501, 50).tolist() + [None])), # Fixed syntax
                "max_depth":    (np.array([5, 15]) if quick 
                     else (np.arange(1, 26, 5).tolist() + [None])),  # Added None correctly
            },

            refine_fn = _refine_rf,
        ),

        ModelSpec(
            name      = "Gradient Boosting (tuned)",
            estimator = GradientBoostingClassifier(random_state=0),
            dataset   = "base",
            param_grid_1 = {
                "learning_rate": ([0.05, 0.1]   if quick else [0.01, 0.05, 0.1]),
                "n_estimators":  ([50, 100]      if quick else [50, 100, 200, 300, 500, 600, 700, 800, 900, 1000]),
                "max_depth":     ([3]            if quick else [1, 2, 3, 5, 7, 10]),
            },
            refine_fn = _refine_gbt,
        ),

        # Nystroem RBF approximation — true kernel SVM at O(n x n_components) cost.
        # Round 1: gamma(8) x n_components(3) x C(8) = 192 combos
        # Round 2: 9 x 9 x 1 = 81 combos (+-0.75 log-decades around R1 best)
        ModelSpec(
            name      = "Kernel SVM (Nystroem RBF)",
            estimator = _kernel_svm_pipeline(),
            dataset   = "base",
            param_grid_1 = {
                "nystroem__gamma":        (np.logspace(-2, 0, 2) if quick else
                                           np.logspace(-4, 3, 8)),
                "nystroem__n_components": ([200] if quick else [200, 500, 1000]),
                "clf__C":                 (np.logspace(-1, 1, 2) if quick else
                                           np.logspace(-3, 4, 8)),
            },
            refine_fn = _refine_kernel_svm,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 5 — EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    """
    Convert a model display name to a safe filename stem.
    'Random Forest (tuned)' -> 'Random_Forest_tuned'
    """
    stem = re.sub(r"[^\w\s-]", "", name)
    stem = re.sub(r"\s+", "_", stem.strip())
    return stem


def _write_readme(
    out: pathlib.Path,
    config: Dict[str, Any],
    data: PreparedData,
    result: CompareResult,
    elapsed_sec: float,
    log_filename: str,
) -> None:
    """
    Write README.md into the output directory.

    Sections
    --------
    1. Run metadata      (timestamp, target, quick mode, wall time)
    2. Target definition (class table with balanced weights)
    3. Dataset summary   (raw shape, train/test sizes)
    4. Dropped columns   (with rationale)
    5. Features used     (numeric + categorical, after dropping & before OHE)
    6. Feature engineering summary
    7. Model comparison  (ranked by test accuracy, Markdown table)
    8. Files in this directory
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── 2. Target description ─────────────────────────────────────────────────
    if config["target_mode"] == "4class":
        target_desc = (
            "**Mode**: 4-class (`application_result`)\n\n"
            "Synthesised from three raw binary columns:\n\n"
            "| Value | Label | Meaning |\n"
            "|------:|-------|----------|\n"
            "| 0 | No Outcome | No interviews, no offer |\n"
            "| 1 | First Round Only | First-round interview only |\n"
            "| 2 | Second Round | Reached second round, no offer |\n"
            "| 3 | Offer Received | Job offer received |\n"
        )
    else:
        col = config.get("target_col", "?")
        target_desc = (
            f"**Mode**: Binary (`{col}`)\n\n"
            f"Raw column `{col}` used directly as the 0/1 target."
        )

    # ── Class distribution table ──────────────────────────────────────────────
    train_dist = data.y_train.value_counts().sort_index()
    test_dist  = data.y_test.value_counts().sort_index()

    dist_rows = []
    for cls in sorted(data.label_map):
        lbl  = data.label_map[cls]
        tr   = int(train_dist.get(cls, 0))
        te   = int(test_dist.get(cls, 0))
        w    = data.class_weights.get(cls, float("nan"))
        pct  = 100 * tr / max(len(data.y_train), 1)
        dist_rows.append(
            f"| {cls} | {lbl} | {tr:,} ({pct:.1f}%) | {te:,} | {w:.4f} |"
        )
    dist_table = "\n".join(dist_rows)

    # ── 4. Dropped columns ────────────────────────────────────────────────────
    drop_lines = "\n".join(f"- `{c}`" for c in config["drop_cols"])

    # ── 5. Features ───────────────────────────────────────────────────────────
    num_lines = "\n".join(f"- `{c}`" for c in data.numeric_cols)
    cat_lines = "\n".join(f"- `{c}`" for c in data.categorical_cols)

    # ── 6. Feature engineering ────────────────────────────────────────────────
    if config.get("do_fe") and data.poly is not None:
        selected_str = (
            ", ".join(f"`{c}`" for c in data.selected_fe_cols)
            or "*(none selected)*"
        )
        fe_block = (
            f"- **Enabled**: yes\n"
            f"- **Degree**: {config['fe_degree']}  |  "
            f"interaction-only: `{config['fe_interaction_only']}`\n"
            f"- **Selector estimator**: `{config['fe_kind']}`\n"
            f"- **Candidate polynomial columns**: {len(data.poly_candidate_cols)}\n"
            f"- **Selected polynomial columns**: {len(data.selected_fe_cols)}\n"
            f"  {selected_str}\n"
        )
    else:
        fe_block = "- **Enabled**: no (skipped)\n"

    # ── 7. Model comparison table ─────────────────────────────────────────────
    df = result.comparison_df.copy()
    df["Test Accuracy"] = pd.to_numeric(df["Test Accuracy"], errors="coerce")

    md_rows = [
        "| Rank | Model | Dataset | Val Acc R1 | Val Acc R2 | Test Acc | Time (s) |",
        "|-----:|-------|---------|----------:|----------:|---------:|---------:|",
    ]
    for rank, row in df.iterrows():
        marker   = " **\u25c4**" if row["Model"] == result.best_model_name else ""
        test_acc = (f"{row['Test Accuracy']:.4f}"
                    if not pd.isna(row["Test Accuracy"]) else "—")
        md_rows.append(
            f"| {rank} | {row['Model']}{marker} | {row['Dataset']} |"
            f" {row['Val Acc (R1)']} | {row['Val Acc (R2)']} |"
            f" {test_acc} | {row['Time (s)']} |"
        )
    model_table = "\n".join(md_rows)

    # ── 8. Files in directory ─────────────────────────────────────────────────
    file_rows = [
        f"- `{log_filename}` — full console log; each run is appended with a timestamp header",
        "- `comparison.csv` — ranked model comparison table (machine-readable)",
        "- `README.md` — this file (regenerated on every run)",
    ]
    for name in result.models:
        file_rows.append(
            f"- `{_safe_filename(name)}.joblib` — fitted estimator for **{name}**"
        )
    files_block = "\n".join(file_rows)

    # ── Assemble final markdown ───────────────────────────────────────────────
    md = f"""\
# Export README — Will You Get a Job?

> Auto-generated by `train_and_export.py` on {ts}

---

## 1. Run Metadata

| Key | Value |
|-----|-------|
| Generated at | {ts} |
| Data file | `{config['data_path']}` |
| Target mode | `{config['target_mode']}` |
| Random state | {config['random_state']} |
| Train / test split | {int((1 - config['test_size']) * 100)} / {int(config['test_size'] * 100)} % |
| CV folds | {config['cv']} |
| Scoring metric | `{config['scoring']}` |
| Quick mode | {config.get('_quick_mode', False)} |
| Total wall time | {elapsed_sec / 60:.1f} min |

---

## 2. Target

{target_desc}

### Class distribution

| Class | Label | Train (%) | Test | Balanced weight |
|------:|-------|----------:|-----:|----------------:|
{dist_table}

---

## 3. Dataset Summary

| | Value |
|-|-------|
| Raw CSV rows | {len(data.y_train) + len(data.y_test):,} |
| Training records | {len(data.y_train):,} |
| Test records | {len(data.y_test):,} |
| Base features (after OHE) | {len(data.feature_names)} |

---

## 4. Dropped Columns

The following columns were removed **before** any feature engineering.

{drop_lines}

**Rationale**:
- `Student_ID` — pure identifier, carries zero predictive signal.
- `Time_to_Offer_Days` — over 60,000 NaN values; also a post-outcome leaker (only known after an offer exists).
- `Offer_Salary`, `Company_Size_Offered`, `Role_Relevance` — known only after receiving an offer (data leakage); also >60% missing.
- `Accepted_Offer` — directly downstream of `Offer_Received` (data leakage).

---

## 5. Features Used

**Total base features after one-hot encoding**: {len(data.feature_names)}
({len(data.numeric_cols)} numeric + {len(data.categorical_cols)} categorical groups)

### Numeric features — standardised with `StandardScaler`

{num_lines}

### Categorical features — one-hot encoded (`drop_first=True`)

{cat_lines}

---

## 6. Feature Engineering

{fe_block}

---

## 7. Model Comparison

{model_table}

---

## 8. Files in This Directory

{files_block}
"""

    readme_path = out / "README.md"
    readme_path.write_text(md, encoding="utf-8")
    print(f"  saved  README.md")


def _export_all(
    out: pathlib.Path,
    config: Dict[str, Any],
    data: PreparedData,
    result: CompareResult,
    elapsed_sec: float,
    log_filename: str,
) -> None:
    """
    Persist all run artefacts to the output directory:
      - one .joblib file per fitted model (best-to-worst order)
      - comparison.csv
      - README.md
    (run.log is already written by the Tee installed in run())
    """
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[export] Writing outputs to  '{out.resolve()}' ...")

    # ── Fitted models ─────────────────────────────────────────────────────────
    for name, estimator in result.models.items():
        fname = out / f"{_safe_filename(name)}.joblib"
        joblib.dump(estimator, fname)
        print(f"  saved  {fname.name}")

    # ── Comparison table ──────────────────────────────────────────────────────
    result.comparison_df.to_csv(out / "comparison.csv")
    print(f"  saved  comparison.csv")

    # ── README ────────────────────────────────────────────────────────────────
    _write_readme(out, config, data, result, elapsed_sec, log_filename)

    print("[export] Done.")


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 6 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(config: Dict[str, Any]) -> None:
    """
    End-to-end pipeline:

    1. Resolve output directory name from config (target_mode + target_col).
    2. Create output dir; install Tee logger (stdout + stderr -> run.log).
    3. Load raw CSV.
    4. prepare(df, config)           -> PreparedData
    5. _build_project_specs(config)  -> List[ModelSpec]
    6. train_and_compare(...)        -> CompareResult
    7. _export_all(...)              -> .joblib files + comparison.csv + README.md
    8. Remove Tee logger (always, even on exception).
    """
    t_total = time.perf_counter()

    # ── 1 & 2. Resolve output dir and start logging ───────────────────────────
    out_dir      = _resolve_output_dir(config)
    log_filename = "run.log"
    log_path     = out_dir / log_filename
    out_dir.mkdir(parents=True, exist_ok=True)

    tee_out, tee_err = _install_tee(log_path)

    try:
        # ── 3. Load ───────────────────────────────────────────────────────────
        data_path = config["data_path"]
        print(f"\n[main] Loading data from '{data_path}' ...")
        if not pathlib.Path(data_path).exists():
            raise FileNotFoundError(
                f"Data file not found: '{data_path}'. "
                "Pass the correct path via --data or update CONFIG['data_path']."
            )
        df = pd.read_csv(data_path)
        print(f"[main] Raw shape: {df.shape}")

        # ── 4. Prepare ────────────────────────────────────────────────────────
        print("\n[main] Preparing data ...")
        data = prepare(df, config)

        # ── 5. Model specs ────────────────────────────────────────────────────
        print("\n[main] Building model specs ...")
        specs = _build_project_specs(config)
        print(f"[main] {len(specs)} models will be trained:")
        for s in specs:
            print(f"  * {s.name}  [{s.dataset}]")

        # ── 6. Train & compare ────────────────────────────────────────────────
        print("\n[main] Training models ...")
        compare_result = train_and_compare(data, specs, config)

        elapsed = time.perf_counter() - t_total
        print(f"\n[main] Total wall time: {elapsed / 60:.1f} min")

        # ── 7. Export ─────────────────────────────────────────────────────────
        _export_all(out_dir, config, data, compare_result, elapsed, log_filename)

    finally:
        # ── 8. Always restore stdout/stderr (even if training crashed) ────────
        _remove_tee(tee_out, tee_err)

    print(f"\n[main] All outputs written to: {out_dir.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
#   SECTION 7 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train all models, export pipelines, write README.md and run.log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default=None, metavar="PATH",
        help="Path to raw CSV.  Overrides CONFIG['data_path'].",
    )
    p.add_argument(
        "--output-dir-base", default=None, metavar="PREFIX",
        help="Base prefix for output dir.  Overrides CONFIG['output_dir_base'].",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=(
            "Tiny coarse grids for fast dev iteration (~1 min total). "
            "Results are NOT representative of final performance."
        ),
    )
    p.add_argument(
        "--no-fe", action="store_true",
        help="Skip polynomial feature engineering (faster; weaker linear models).",
    )
    p.add_argument(
        "--target", choices=["4class", "binary"], default=None,
        help=(
            "'4class' synthesises application_result {0-3}. "
            "'binary' uses --target-col directly."
        ),
    )
    p.add_argument(
        "--target-col", default=None, metavar="COL",
        help="Binary target column name (only with --target binary).",
    )
    p.add_argument(
        "--cv", type=int, default=None, metavar="N",
        help="Number of cross-validation folds.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Shallow copy — never mutate the module-level CONFIG dict
    cfg = dict(CONFIG)

    if args.data            is not None: cfg["data_path"]       = args.data
    if args.output_dir_base is not None: cfg["output_dir_base"] = args.output_dir_base
    if args.target          is not None: cfg["target_mode"]     = args.target
    if args.target_col      is not None: cfg["target_col"]      = args.target_col
    if args.cv              is not None: cfg["cv"]              = args.cv
    if args.quick:                       cfg["_quick_mode"]      = True
    if args.no_fe:                       cfg["do_fe"]            = False

    out_dir = _resolve_output_dir(cfg)

    print("=" * 62)
    print("  Will You Get a Job? -- train_and_export")
    print("=" * 62)
    print(f"  target_mode  : {cfg['target_mode']}")
    if cfg["target_mode"] == "binary":
        print(f"  target_col   : {cfg['target_col']}")
    print(f"  output_dir   : {out_dir}")
    print(f"  log file     : {out_dir / 'run.log'}")
    print(f"  quick_mode   : {cfg['_quick_mode']}")
    print(f"  do_fe        : {cfg['do_fe']}")
    print(f"  cv           : {cfg['cv']}")
    print("=" * 62)

    run(cfg)