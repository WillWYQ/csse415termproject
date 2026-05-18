"""
prepare.py
==========
Handles all data preparation for the job-search prediction project.

Responsibilities
----------------
1. Clean          – drop columns listed in config['drop_cols'] only
2. Target         – build a binary or 4-class target from the raw columns
3. Split          – stratified 80/20 train/test split
4. Scale          – StandardScaler fitted on training numeric columns only
5. Feature Eng.   – optional polynomial + greedy-forward-selection step

Public API
----------
    result = prepare(df, config)          # main entry point

  Training data
    result.X_train_FE                     # feature-engineered train set
    result.X_test_FE                      # feature-engineered test set
    result.X_train_base                   # scaled, pre-engineering train set
    result.X_test_base                    # scaled, pre-engineering test set
    result.y_train / result.y_test

  Interpretation (not used in training)
    result.features                       # all feature names before one-hot encoding
    result.numeric_cols                   # numeric feature names
    result.categorical_cols               # categorical feature names (tells caller
                                          # which columns were one-hot encoded, since
                                          # originals are replaced by dummy columns)
    result.label_map                      # int → human-readable class name
    result.n_classes                      # number of distinct classes
    result.class_weights                  # {class: weight} balanced weights for train set

  Fitted objects (needed for inference / reporting)
    result.scaler                         # fitted StandardScaler
    result.poly                           # fitted PolynomialFeatures (or None)
    result.selected_fe_cols               # polynomial cols kept by greedy FE
    result.poly_candidate_cols            # all polynomial cols before selection
    result.feature_names                  # column order of X_train_base

Called by train_and_export.py, which owns all config.

AI Usage
--------
This file was developed with the assistance of generative AI based on our previous work.
We ensured we understand and verify all AI-generated code, and we are responsible for its correctness and suitability for our project.
"""

from __future__ import annotations

import os

import joblib

# ── Thread-count control (set before numpy / sklearn are imported) ────────────
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

n_physical = joblib.cpu_count(only_physical_cores=True)
N_JOBS = max(1, n_physical - 2)

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL.ImagePalette import raw
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Return container
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """All artefacts produced by prepare(). Passed straight to the trainers."""

    # ---- training data -------------------------------------------------- #

    # Feature-engineered sets — for linear / distance-based models:
    #   LogReg, Ridge, Lasso, LinearSVC  (fe_kind='logreg')
    #   KNN                              (fe_kind='knn')
    X_train_FE: pd.DataFrame
    X_test_FE:  pd.DataFrame

    # Pre-engineering sets — for models that capture non-linearity internally:
    #   Tree models  (DT, RF, GBT)      — splits already discover interactions
    #   Kernel SVM   (RBF / poly)       — kernel trick handles the mapping;
    #                                     adding poly features on top changes
    #                                     the effective kernel uncontrollably
    X_train_base: pd.DataFrame
    X_test_base:  pd.DataFrame

    y_train: pd.Series
    y_test:  pd.Series

    # ---- interpretation (not used in training) -------------------------- #

    # All feature names as they existed before one-hot encoding.
    # Useful for reports and feature-importance plots that should reference
    # the original column names, not the expanded dummy names.
    features: List[str]

    # Names of numeric and categorical columns separately.
    numeric_cols:     List[str]
    categorical_cols: List[str]   # tells caller which columns were one-hot
                                  # encoded, since originals are replaced by
                                  # dummy columns in X_train_base / X_train_FE

    label_map:     Dict[int, str]  # int → human-readable class name
    n_classes:     int             # number of distinct classes
    class_weights: Dict[int, float]  # balanced weights computed on train set;
                                     # pass to models that accept class_weight

    # ---- fitted objects ------------------------------------------------- #

    scaler:             StandardScaler
    poly:               Optional[PolynomialFeatures]  # None when FE skipped
    selected_fe_cols:   List[str]       # polynomial cols kept after greedy FE
    poly_candidate_cols: List[str]      # all polynomial cols before selection
    feature_names:      List[str]       # column order of X_train_base


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------

def _build_binary_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Return a strict 0 / 1 Series for the given target column.

    If the column already contains only {0, 1} values it is returned as-is
    (fast path for Offer_Received).  Otherwise every value > 0 is mapped to 1
    and 0 stays 0.

    This handles raw count columns (First_Round_Interviews, Second_Round_Interviews)
    which store the number of interviews held, not a binary flag.  Binarising
    gives the natural interpretation: "did the student get any interview?".
    It also prevents stratified train_test_split from failing on rare count
    values that have only one sample.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Binary target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    raw = df[target_col]
    unique_vals = set(raw.dropna().unique())

    if unique_vals <= {0, 1}:
        # Already strictly binary — fast path (e.g. Offer_Received)
        return raw.astype(int)

    # Count column: binarise by positivity  (0 → 0, any positive count → 1)
    binarised = (raw > 0).astype(int)
    n_pos   = int(binarised.sum())
    n_total = len(binarised)
    sorted_uniq = sorted(unique_vals)
    preview = sorted_uniq[:8]
    ellipsis = "…" if len(sorted_uniq) > 8 else ""
    print(
        f"  [prepare] '{target_col}' is a count column "
        f"(unique values: {preview}{ellipsis}).\n"
        f"  Binarised to (>0)=1: {n_pos:,} positives / {n_total:,} total "
        f"({100 * n_pos / n_total:.1f}%)"
    )
    return binarised


def _build_4class_target(df: pd.DataFrame) -> pd.Series:
    """
    Synthesise a 4-class label from the three raw outcome columns:
        0 = no interview, no offer
        1 = first-round interview only
        2 = reached second round, no offer
        3 = offer received

    The three source columns must already be present in df — do not drop them
    in config['drop_cols'] when using 4class mode.
    """
    required = ["First_Round_Interviews", "Second_Round_Interviews", "Offer_Received"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"4-class target needs {required}; missing from df: {missing}. "
            "Remove these from config['drop_cols']."
        )

    def _label(row: pd.Series) -> int:
        if row["Offer_Received"] == 1:
            return 3
        if row["Second_Round_Interviews"] > 0:
            return 2
        if row["First_Round_Interviews"] > 0:
            return 1
        return 0

    return df.apply(_label, axis=1)


# ---------------------------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------------------------

def _make_fe_estimator(fe_kind: str, random_state: int, max_iter: int):
    """
    Return an unfitted sklearn estimator used inside the greedy forward
    selector.

    fe_kind options
    ---------------
    'logreg'  – LogisticRegression (default).
                Selects features that improve a linear decision boundary;
                ideal for LogReg, Ridge, Lasso, LinearSVC.

    'knn'     – KNeighborsClassifier.
                Selects features that compress class overlap in Euclidean
                space; ideal for KNN itself.

    Note: kernel SVM and tree-based models always use X_train_base and
    never call this function.
    """
    if fe_kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            fit_intercept=False,
            max_iter=max_iter,
            random_state=random_state,
        )
    if fe_kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        # k=11 is more stable than the default 5 on large datasets
        return KNeighborsClassifier(n_neighbors=11, n_jobs=N_JOBS)

    raise ValueError(
        f"Unknown fe_kind '{fe_kind}'. Supported values: 'logreg', 'knn'."
    )


def _greedy_forward_select(
    X_poly_train: pd.DataFrame,
    y_train: pd.Series,
    estimator,
    cv: int = 3,
) -> List[str]:
    """
    Greedy forward feature selection over polynomial / interaction columns.

    Starts from a bias column and iteratively appends the candidate that
    yields the highest mean cross-validated accuracy.  Tracks the globally
    best prefix and returns its column names (excluding 'bias').

    Parameters
    ----------
    X_poly_train : DataFrame with a 'bias' column of 1s plus all candidates
    y_train      : target Series
    estimator    : unfitted sklearn classifier used for CV scoring
    cv           : number of CV folds (3 keeps runtime reasonable on 80k rows)
    """
    model: List[str] = ["bias"]
    candidates: List[str] = [c for c in X_poly_train.columns if c != "bias"]

    # Baseline: bias-only score
    cv_res = cross_validate(
        estimator,
        X_poly_train[model].values.reshape(-1, 1),
        y_train,
        scoring="accuracy",
        cv=cv,
        n_jobs=N_JOBS,
    )
    best_accuracy = float(cv_res["test_score"].mean())
    best_model = model.copy()

    while candidates:
        print(f"  [FE] remaining candidates: {len(candidates)}", flush=True)

        scores = np.zeros(len(candidates))
        for k, cand in enumerate(candidates):
            cv_res = cross_validate(
                estimator,
                X_poly_train[model + [cand]],
                y_train,
                scoring="accuracy",
                cv=cv,
                n_jobs=N_JOBS,
            )
            scores[k] = float(cv_res["test_score"].mean())

        best_k   = int(scores.argmax())
        best_acc = float(scores[best_k])
        chosen   = candidates.pop(best_k)
        model.append(chosen)

        print(f"  [FE] +{chosen}  →  cv acc = {best_acc:.4f}", flush=True)

        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_model = model.copy()

    return [c for c in best_model if c != "bias"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, config: dict) -> PreparedData:
    """
    Run the full data-preparation pipeline.

    Parameters
    ----------
    df     : raw DataFrame loaded from CSV
    config : dict produced by train_and_export.py.  Expected keys:

        drop_cols           : List[str]  columns to drop before anything else.
                                         Only these columns are dropped — no
                                         outcome columns are removed implicitly.
                                         Caller is responsible for including
                                         any leakage columns here.
        target_mode         : str        'binary' | '4class'
        target              : str        which column is the binary target
                                         (only used when target_mode='binary')
        test_size           : float      fraction for test split (default 0.2)
        random_state        : int        (default 0)
        feature_engineering : bool       whether to run greedy poly FE step
        fe_degree           : int        polynomial degree (default 2)
        fe_interaction_only : bool       (default False)
        fe_cv_folds         : int        CV folds for greedy selection (default 3)
        fe_lr_max_iter      : int        max_iter for the internal LR (default 500)
        fe_kind             : str        which model drives feature selection
                                         'logreg' (default) | 'knn'

    Returns
    -------
    PreparedData
    """

    print("[prepare] Starting data preparation …")
    # ------------------------------------------------------------------ #
    # 0.  Resolve config with safe defaults
    # ------------------------------------------------------------------ #
    drop_cols:           List[str] = config.get("drop_cols", [])
    target_mode:         str       = config.get("target_mode", "binary")
    target_col:          str       = config.get("target", "Offer_Received")
    test_size:           float     = config.get("test_size", 0.2)
    random_state:        int       = config.get("random_state", 0)
    do_fe:               bool      = config.get("feature_engineering", True)
    fe_degree:           int       = config.get("fe_degree", 2)
    fe_interaction_only: bool      = config.get("fe_interaction_only", False)
    fe_cv_folds:         int       = config.get("fe_cv_folds", 3)
    fe_lr_max_iter:      int       = config.get("fe_lr_max_iter", 500)
    fe_kind:             str       = config.get("fe_kind", "logreg")

    # ------------------------------------------------------------------ #
    # 1.  Clean — drop only the columns listed in config['drop_cols']
    # ------------------------------------------------------------------ #
    print("[prepare] Step 1 — cleaning …")
    df_clean = df.drop(columns=drop_cols, errors="ignore").copy()

    n_dups = int(df_clean.duplicated().sum())
    if n_dups:
        print(f"  Dropping {n_dups} duplicate rows.")
        df_clean = df_clean.drop_duplicates()

    print(f"  Shape after cleaning: {df_clean.shape}")

    # ------------------------------------------------------------------ #
    # 2.  Build target
    # ------------------------------------------------------------------ #
    print("[prepare] Step 2 — building target …")

    if target_mode == "binary":
        y = _build_binary_target(df_clean, target_col)
        label_map: Dict[int, str] = {0: f"No {target_col}", 1: target_col}
    elif target_mode == "4class":
        y = _build_4class_target(df_clean)
        label_map = {
            0: "No Interview",
            1: "1st Round Only",
            2: "2nd Round, No Offer",
            3: "Offer Received",
        }
    else:
        raise ValueError(
            f"Unknown target_mode '{target_mode}'. Use 'binary' or '4class'."
        )

    print(f"  Class distribution:\n{y.value_counts().sort_index().to_string()}")

    # Remove the target column from X (caller controls everything else via
    # drop_cols — no implicit leakage prevention beyond the target itself)
    X = df_clean.drop(columns=[target_col], errors="ignore")

    # ------------------------------------------------------------------ #
    # 3.  Separate numeric / categorical; record pre-encoding names;
    #     one-hot encode categoricals
    # ------------------------------------------------------------------ #
    print("[prepare] Step 3 — encoding …")

    X_numeric      = X.select_dtypes(include=[np.number])
    X_categorical   = X.select_dtypes(include=["object", "string"])
    numeric_cols    = list(X_numeric.columns)
    categorical_cols = list(X_categorical.columns)

    # features = all column names before encoding, in original order.
    # Useful for interpretation and reports; not used in training.
    features = list(X.columns)

    print(f"  Numeric     ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical ({len(categorical_cols)}): {categorical_cols}")

    X_cat_encoded = pd.get_dummies(X_categorical, drop_first=True)

    if len(X_cat_encoded.columns) > 50:
        warnings.warn(
            f"One-hot encoding produced {len(X_cat_encoded.columns)} columns. "
            "Consider dropping high-cardinality features via drop_cols.",
            UserWarning,
            stacklevel=2,
        )

    X_full = pd.concat([X_numeric, X_cat_encoded], axis=1)

    # ------------------------------------------------------------------ #
    # 4.  Stratified train / test split
    # ------------------------------------------------------------------ #
    print("[prepare] Step 4 — train/test split …")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"  Train {X_train.shape}  |  Test {X_test.shape}")

    # Compute balanced class weights from the training set only.
    # Models that accept class_weight (LogReg, SVC, DT, RF …) can use
    # result.class_weights directly without recomputing.
    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights: Dict[int, float] = dict(zip(classes.tolist(), weights.tolist()))
    n_classes = len(classes)

    print(f"  n_classes={n_classes}  class_weights={class_weights}")

    # ------------------------------------------------------------------ #
    # 5.  StandardScaler — fit on train only, apply to both splits
    # ------------------------------------------------------------------ #
    print("[prepare] Step 5 — standardising …")

    scaler = StandardScaler()

    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols,
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test[numeric_cols]),
        columns=numeric_cols,
    )

    X_train_scaled = pd.concat(
        [X_train_num, X_train.drop(columns=numeric_cols).reset_index(drop=True)],
        axis=1,
    )
    X_test_scaled = pd.concat(
        [X_test_num, X_test.drop(columns=numeric_cols).reset_index(drop=True)],
        axis=1,
    )

    # Pre-FE snapshot.
    #   Tree models (DT, RF, GBT): splits discover interactions internally;
    #     polynomial expansion adds nothing and inflates training time.
    #   Kernel SVM (RBF / poly): the kernel maps inputs to high-dimensional
    #     space implicitly; applying polynomial features beforehand changes
    #     the effective kernel in an uncontrolled way and typically degrades
    #     performance.
    X_train_base = X_train_scaled.copy()
    X_test_base  = X_test_scaled.copy()
    feature_names = list(X_train_base.columns)

    # ------------------------------------------------------------------ #
    # 6.  Feature engineering — polynomial expansion + greedy forward
    #     selection driven by fe_kind.
    #
    #     Linear / distance models that benefit from explicit non-linear
    #     features use X_train_FE:
    #       fe_kind='logreg' → LogReg, Ridge, Lasso, LinearSVC
    #       fe_kind='knn'    → KNN
    # ------------------------------------------------------------------ #
    selected_fe_cols:    List[str] = []
    poly_candidate_cols: List[str] = []
    poly: Optional[PolynomialFeatures] = None

    if do_fe:
        print(
            f"[prepare] Step 6 — feature engineering "
            f"(degree={fe_degree}, fe_kind='{fe_kind}') …"
        )

        poly = PolynomialFeatures(
            degree=fe_degree,
            interaction_only=fe_interaction_only,
            include_bias=False,
        )

        # Expand numeric columns only — polynomial terms of binary one-hot
        # columns are redundant (x² = x for x ∈ {0, 1}).
        
        X_train_poly_arr = poly.fit_transform(X_train_scaled[numeric_cols])
        poly_candidate_cols = list(poly.get_feature_names_out())
        X_train_poly = pd.DataFrame(X_train_poly_arr, columns=poly_candidate_cols)

        
        X_test_poly = pd.DataFrame(
            poly.transform(X_test_scaled[numeric_cols]),
            columns=poly_candidate_cols,
        )

        # Bias column: the greedy selector uses it as the null starting model
        X_train_poly["bias"] = 1.0
        X_test_poly["bias"]  = 1.0

        estimator = _make_fe_estimator(fe_kind, random_state, fe_lr_max_iter)

        optimal_cols = _greedy_forward_select(
            X_train_poly, y_train, estimator, cv=fe_cv_folds
        )

        # Retain only columns genuinely new relative to the base feature set
        selected_fe_cols = [
            c for c in optimal_cols if c not in X_train_scaled.columns
        ]
        print(
            f"  [FE] {len(selected_fe_cols)} / {len(poly_candidate_cols)} "
            "polynomial columns selected."
        )

        X_train_FE = pd.concat(
            [
                X_train_scaled.reset_index(drop=True),
                X_train_poly[selected_fe_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        X_test_FE = pd.concat(
            [
                X_test_scaled.reset_index(drop=True),
                X_test_poly[selected_fe_cols].reset_index(drop=True),
            ],
            axis=1,
        )

    else:
        print("[prepare] Step 6 — feature engineering skipped.")
        X_train_FE = X_train_scaled.copy()
        X_test_FE  = X_test_scaled.copy()

    # ------------------------------------------------------------------ #
    # 7.  Final summary
    # ------------------------------------------------------------------ #
    print("\n[prepare] Done.")
    print(f"  X_train_FE   : {X_train_FE.shape}")
    print(f"  X_train_base : {X_train_base.shape}")
    print(f"  Features before encoding ({len(features)}): {features}")

    return PreparedData(
        X_train_FE          = X_train_FE,
        X_test_FE           = X_test_FE,
        X_train_base        = X_train_base,
        X_test_base         = X_test_base,
        y_train             = y_train.reset_index(drop=True),
        y_test              = y_test.reset_index(drop=True),
        features            = features,
        numeric_cols        = numeric_cols,
        categorical_cols    = categorical_cols,
        label_map           = label_map,
        n_classes           = n_classes,
        class_weights       = class_weights,
        scaler              = scaler,
        poly                = poly,
        selected_fe_cols    = selected_fe_cols,
        poly_candidate_cols = poly_candidate_cols,
        feature_names       = feature_names,
    )