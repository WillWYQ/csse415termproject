# MA 415 Final Project — Try to Get a Job

**Group:** Grp Try to get a Job  
**Course:** MA 415, Rose-Hulman Institute of Technology, Spring 2026  
**Contributors:** Yueqiao Wang, Mingkun Fu, Conner Tavares

---

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Dataset](#dataset)
- [Version History](#version-history)
  - [Version 0 — Initial Exploration](#version-0--initial-exploration)
  - [Version 1 — Regularization + Random Forest](#version-1--regularization--random-forest)
  - [Version 2 — Stratified Split + Gradient Boosting](#version-2--stratified-split--gradient-boosting)
  - [Version 3 — Binary Classification Pivot](#version-3--binary-classification-pivot)
  - [Version 4 — Feature Engineering + Full Model Suite](#version-4--feature-engineering--full-model-suite)
  - [!!! Final Submission — Cleaned & Documented](#-final-submission--cleaned--documented)
- [Branch Overview](#branch-overview)
  - [`fum` — Subgroup Analysis Notebooks](#fum--subgroup-analysis-notebooks)
  - [`fum` — Subgroup Analysis Notebooks](#fum--subgroup-analysis-notebooks-1)
  - [`export` — Scriptified Training Pipeline](#export--scriptified-training-pipeline)
  - [`webApp` — Browser-Based Job Offer Predictor](#webapp--browser-based-job-offer-predictor)
  - [Branch Summary](#branch-summary)
- [Quick Reference: Files at a Glance](#quick-reference-files-at-a-glance)
- [Setup](#setup)
- [AI Disclosure](#ai-disclosure)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Dataset

| File | Description |
|------|-------------|
| [`data/job_search_platform_efficacy_100k.csv`](data/job_search_platform_efficacy_100k.csv) | 100,000 student job-search records. Raw features include university rating, school size, region, major, GPA, internships, extracurriculars, networking events, primary search platform, months searching, applications submitted, first/second round interviews, and offer outcome. |

---

## Version History

### Version 0 — Initial Exploration
**File:** [`src/InitialWork.ipynb`](src/InitialWork.ipynb)  
**Cells:** 41

**What this version does:**
- Loads the raw CSV; identifies columns with high missing rates (`Accepted_Offer`, `Role_Relevance`, `Offer_Salary`, etc.)
- Drops leakage/identifier columns and builds a **4-class target variable** `application_result`:
  - `0` — No Interview
  - `1` — First Round Interview, No Offer
  - `2` — Second Round Interview, No Offer
  - `3` — Got Offer
- Preprocessing: numeric/categorical split, one-hot encoding, 80/20 train-test split (no stratify, `random_state=0`), StandardScaler on numeric features
- **19 features** (6 numeric + 13 OHE categorical; interview columns dropped into the target)
- Exploratory plots: feature distributions, correlation heatmap, class distribution

**Models trained:**

| Model | Test Accuracy |
|-------|--------------|
| Baseline (most frequent class) | 34.09% |
| KNN (n=5, default) | 35.93% |
| Logistic Regression | 43.57% |

---

### Version 1 — Regularization + Random Forest
**File:** [`src/InitialWork1.ipynb`](src/InitialWork1.ipynb)  
**Cells:** 58

**Builds on Version 0. What's new:**
- Ridge Logistic Regression with GridSearchCV over α ∈ {0.0001 … 100000} (best α = 0.0001)
- Lasso Logistic Regression with GridSearchCV (best α = 1.0)
- Random Forest with broad GridSearchCV (n_estimators ∈ 1–501, max_depth ∈ 1–26) then fine-tuned narrow search
- Observation: Ridge and Lasso give nearly identical accuracy to unregularized LogReg — regularization strength doesn't matter much for this dataset

**Models trained (additions):**

| Model | Test Accuracy |
|-------|--------------|
| Ridge Logistic Regression | 43.55% |
| Lasso Logistic Regression | 43.55% |
| Random Forest (best: n=401, depth=11) | 44.45% |
| Random Forest fine-tuned (n=404, depth=11) | 44.45% |

---

### Version 2 — Stratified Split + Gradient Boosting
**File:** [`src/V2.ipynb`](src/V2.ipynb)  
**Cells:** 65

**Builds on Version 1. What's new:**
- **Stratified train-test split** (`stratify=y`) to ensure balanced class proportions
- Tuned KNN by using GridSearchCV (n_neighbors 3–31, weights: uniform/distance, metric: euclidean/manhattan)
- Gradient Boosting Classifier added with GridSearchCV (learning_rate, n_estimators, max_depth) + fine-tuned search

**Models trained (additions):**

| Model | Test Accuracy |
|-------|--------------|
| KNN tuned (best: manhattan, n=31, distance weights) | 42.92% |
| Gradient Boosting (best: lr=0.05, depth=3, n=100) | 44.92% |

> Note: This verison still using the 4-class target. Accuracy ceiling ~45% reflects the difficulty of distinguishing interview stages from student background features alone.

---

### Version 3 — Binary Classification Pivot
**File:** [`src/V3.ipynb`](src/V3.ipynb)  
**Cells:** 67

**Key design change:** Switched from the engineered 4-class `application_result` to the raw binary `Offer_Received` (0 = no offer, 1 = offer). Crucially, `First_Round_Interviews` and `Second_Round_Interviews` are now kept as **input features** (not folded into the target), giving **21 features** (8 numeric + 13 OHE categorical).

**Why this matters:** Interview counts are strong predictors of offer — accuracy jumps dramatically. However, we do realized that this is causing potential data leakage during review.

**Models trained (all re-run on binary target):**

| Model | Test Accuracy |
|-------|--------------|
| Baseline (most frequent = no offer) | 65.77% |
| KNN tuned | 87.66% |
| Logistic Regression | 80.18% |
| Ridge Logistic Regression (best α = 10.0) | 80.18% |
| Lasso Logistic Regression (best α = 0.0001) | 80.19% |
| Random Forest (best: n=101, depth=16) | 91.04% |
| Random Forest fine-tuned (n=100, depth=17) | 91.03% |
| Gradient Boosting (best: lr=0.05, depth=5, n=300) | 91.35% |
| Gradient Boosting fine-tuned (n=310) | 91.32% |

---

### Version 4 — Feature Engineering + Full Model Suite
**File:** [`src/V4.ipynb`](src/V4.ipynb)  
**Cells:** 94


**Builds on Version 3. What's new:**
- **Feature engineering:** degree-2 polynomial + interaction terms on the 8 numeric features, creating `X_train_poly` alongside the original `X_train_base`. Tree models use `X_train_base`; linear models use `X_train_poly`.
- **SVM** (trained on `X_train_base`; RBF kernel)
- **Decision Tree** (trained on `X_train_base` for interpretability)
- **Model Comparison** table — all models ranked by test accuracy
- **Feature Importance Comparison** — RF vs GBT feature importances for feature analysis
- **Individual Prediction Analysis** — per-sample prediction breakdown to help us understand the model performance
- **Confusion Matrix** — best model (GBT) vs most interpretable model (Decision Tree) comparison added
- **Section description** AI-generated section description comments with human review added to each major section for clarity and save time when doing rerun for typical model

**Additional models:**

| Model | Notes |
|-------|-------|
| SVM | RBF kernel on `X_train_base` |
| Decision Tree | GridSearchCV on max_depth; fully interpretable |

---

### !!! Final Submission — Cleaned & Documented
**File:** [`src/Final Submission.ipynb`](src/Final%20Submission.ipynb)  
**Cells:** 94

**Identical in content to Version 4** 

This is the **final version** for grading.

---

## Branch Overview

The repository has four branches beyond `main`, each representing a distinct line of work.

TODO:
- [ ] Conner Tavares need to push their commits.

---

### `fum` — Subgroup Analysis Notebooks
> **Human-written** exploratory analysis by Conner Tavares

Two standalone notebooks that investigate specific subpopulations rather than the full dataset.

| File | Cells | What it analyzes |
|------|-------|-----------------|
| [`src/1st_interview.ipynb`](src/1st_interview.ipynb) | # | first interview result from all data  |
| [`src/Sec_1st.ipynb`](src/Sec_1st.ipynb) | # | second interview result from the data who get first interview |

### `fum` — Subgroup Analysis Notebooks
> **Human-written** exploratory analysis by Mingkun Fu

Two standalone notebooks that investigate specific subpopulations rather than the full dataset.

| File | Cells | What it analyzes |
|------|-------|-----------------|
| [`src/2nd_interview.ipynb`](src/2nd_interview.ipynb) | 95 | Filters to students who reached a **second-round interview**, then trains the full model suite on that subgroup to identify what separates offer-getters from second-round-only candidates. Conclusion: prior internships and GPA are the strongest predictors within this group. |
| [`src/Background_offer.ipynb`](src/Background_offer.ipynb) | 100 | Trains on **background features only** (drops all interview/search-behavior columns) to measure how much student background alone predicts offers. Conclusion: GPA, search platform, and university rating show moderate signal; background alone cannot match the full-feature accuracy. |

Both notebooks follow the same full pipeline (binary target, feature engineering, all 7 model types, model comparison, confusion matrix, individual prediction analysis) applied to the filtered/restricted dataset.


---

### `export` — Scriptified Training Pipeline
> **Primarily AI-assisted** (all scripts carry the AI Usage disclosure)

To help us on creating the demo, this branch gather the work we have done from verison 4 and converts the notebook experiments into a reusable, command-line Python pipeline.

These pipeline is used to create model that can be used for demo webApp.

While the work is mainly finished by AI, we look through it and understand the way of working.

| File | Purpose |
|------|---------|
| [`finalscript/prepare.py`](finalscript/prepare.py) | Data prep module: clean → target → split → scale → optional polynomial FE. Exposes a `prepare(df, config)` API used by the scripts below. |
| [`finalscript/train_and_compare.py`](finalscript/train_and_compare.py) | Trains a configurable set of models, runs GridSearchCV, and produces a comparison table. |
| [`finalscript/train_and_export.py`](finalscript/train_and_export.py) | Master orchestrator: load CSV → prepare → train → serialize every model to `.joblib` + write `comparison.csv`, `README.md`, `run.log` under an auto-named output directory. Supports `--quick`, `--no-fe`, `--target`, `--cv` flags. |
| [`finalscript/run_all_configs.py`](finalscript/run_all_configs.py) | Batch sweeps multiple experiment configs in one shot: **Group A** (background only, no behavioral features), **Group B** (full features), **Group C** (cascade — earlier-stage outcome used as feature for later-stage target). |

**Experiment groups in `run_all_configs.py`:**

| Group | Target | Features kept | Notes |
|-------|--------|--------------|-------|
| A — restricted | `Offer_Received` | Background only | Drop all search-behavior columns |
| B — full | `Offer_Received` | All features | Background + platform + effort |
| C1 — cascade 2nd (with behavior) | `Second_Round_Interviews` | Background + behavior + First Round | Cascade: first-round count predicts second-round |
| C2 — cascade 2nd (restricted) | `Second_Round_Interviews` | Background + First Round | No behavioral features |
| C3 — cascade offer (with behavior) | `Offer_Received` | Background + behavior + both interview rounds | Strongest cascade setup |
| C4 — cascade offer (restricted) | `Offer_Received` | Background + both interview rounds | No behavioral features |

---


### `webApp` — Browser-Based Job Offer Predictor
> **Primarily AI-assisted** (all scripts carry the AI Usage disclosure)

Extends `export` with two new export scripts and a complete React web application that runs models directly in the browser via ONNX runtime (no backend required).

**New scripts vs `export` branch:**

| File | Purpose |
|------|---------|
| [`finalscript/export_to_js.py`](finalscript/export_to_js.py) | Converts the trained GBT offer-prediction model to ONNX and writes `predictweb/public/model_offer.onnx` + `predictweb/src/lib/preprocessing_offer.js` (scaler params, OHE mapping, form metadata). |
| [`finalscript/export_2nd_to_js.py`](finalscript/export_2nd_to_js.py) | Same for the C1-cascade Random Forest (second-round predictor) → `model_2nd.onnx` + `preprocessing_2nd.js`. |

**Pre-trained model files** in `models/` (one subdirectory per experiment config, each containing `.joblib` files for every model + `comparison.csv` + `run.log`):
- `baseModelOffer/` — baseline binary offer model
- `exported_models__C1_…` through `exported_models__C4_…` — all four cascade configs

**Web app (`predictweb/`)** — Next.js 14 + TypeScript + Tailwind + Shadcn UI:

| Component | Role |
|-----------|------|
| `app/page.tsx` | Landing page with navigation |
| `components/OfferForm.tsx` | Multi-field form for offer prediction (inputs → ONNX inference → result) |
| `components/SecondRoundForm.tsx` | Same for second-round interview prediction |
| `components/ResultCard.tsx` | Displays prediction probability and classification |
| `components/Navbar.tsx` | Site navigation |
| `components/ui/` | Shadcn UI primitives (3D card, globe, floating dock, encrypted text, etc.) |

**CI/CD:** `.github/workflows/deploy-predictweb.yml` deploys the app automatically on push to `webApp`.

**To run locally:**
```bash
cd predictweb
npm install
npm run dev       # http://localhost:3000
```

---

### Branch Summary

| Branch | Authorship | What it adds |
|--------|-----------|-------------|
| `main` | Human | Final notebooks (same as this branch) Major and most important part |
| `export` | AI-assisted | `finalscript/` — scriptified training pipeline with multi-config batch sweep |
| `fum` | Human | `src/2nd_interview.ipynb`, `src/Background_offer.ipynb` — subgroup deep-dives |
| `webApp` | AI-assisted | ONNX model export + Next.js browser predictor app + pre-trained model artifacts |

---

## Quick Reference: Files at a Glance

| File | Cells | Target | Features | Best Accuracy | New in this version |
|------|-------|--------|----------|---------------|---------------------|
| [`InitialWork.ipynb`](src/InitialWork.ipynb) | 41 | 4-class | 19 | 43.57% (LogReg) | Baseline setup, KNN, LogReg |
| [`InitialWork1.ipynb`](src/InitialWork1.ipynb) | 58 | 4-class | 19 | 44.45% (RF) | Ridge, Lasso, Random Forest |
| [`V2.ipynb`](src/V2.ipynb) | 65 | 4-class | 19 | 44.92% (GBT) | Stratified split, Gradient Boosting |
| [`V3.ipynb`](src/V3.ipynb) | 67 | binary | 21 | 91.35% (GBT) | Binary target, interview features kept |
| [`V4.ipynb`](src/V4.ipynb) | 94 | binary | 21 + poly | 91.35% (GBT) | Feature engineering, SVM, DT, analysis |
| [`Final Submission.ipynb`](src/Final%20Submission.ipynb) | 94 | binary | 21 + poly | 91.35% (GBT) | Section comments, final cleanup |

---

## Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Then open any notebook in Jupyter:

```bash
jupyter notebook src/
```

**Key dependencies** (see [`requirements.txt`](requirements.txt)):
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `xgboost`, `shap`, `scipy`, `mlflow`

---

## AI Disclosure

Section-level comments in `Final Submission.ipynb` and `V4.ipynb` were generated with AI assistance for documentation clarity.

`export` and `webApp` were generated with AI assistance for demo purpose under human design and plan.

Data collection in this readme is done by AI due to the repeating work nature.

**All model code, design decisions, and analysis were written by the team.**
