# NeoStat Credit Risk Intelligence Platform

An end-to-end credit risk system built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Given a loan applicant's financial profile, the platform predicts their probability of defaulting, explains *why* that prediction was made, flags breaches of lending policy, and lets you query the dataset in plain English.

> **Tech stack** — Python 3.11, Streamlit, XGBoost / LightGBM (via FLAML AutoML), SHAP, SQLite, Google Gemini, Docker.

---

## Table of Contents

1. [What the Platform Does](#what-the-platform-does)
2. [Architecture Overview](#architecture-overview)
3. [Setup and Running the App](#setup-and-running-the-app)
4. [Model Selection and Class Imbalance Strategy](#model-selection-and-class-imbalance-strategy)
5. [Evaluation Metrics and Results](#evaluation-metrics-and-results)
6. [Prompt Engineering and Token Optimisation](#prompt-engineering-and-token-optimisation)
7. [Rule Derivation Logic and Sample Outputs](#rule-derivation-logic-and-sample-outputs)
8. [Known Limitations and Possible Improvements](#known-limitations-and-possible-improvements)
9. [Repository Structure](#repository-structure)

---

## What the Platform Does

The platform is organised into four user-facing sections, each mapping to an assignment module:

| UI Tab | What it does |
| :--- | :--- |
| **EDA** | Interactive Plotly charts exploring demographics, bureau scores, repayment behaviour, and portfolio quality across 307 511 loan applications. |
| **Risk Prediction** | Enter applicant details, receive a calibrated credit risk score (0-1000), a risk band (Low / Medium / High / Rule Reject), and a SHAP breakdown of the top factors that pushed the risk up or down. |
| **Business Rules** | Shows hard lending-policy rules (instant rejections for extreme cases) alongside automatically mined IF-THEN decision rules extracted from the training data. |
| **Talk to Data** | Type a plain-English question about the dataset and receive a readable answer backed by the SQL query that was run. |

---

## Architecture Overview

The diagram below shows how data flows through the system.

```
Raw CSV files  -->  DataLoader  -->  Merged Parquet Cache (307k x 467 features)
                       |                         |
                       v                         v
                  SQLite DB             DataPreprocessor
               (credit_risk.db)         (clean, impute, encode)
                       |                         |
                       |                         v
         Talk-to-Data  |             FLAML AutoML training
         (Gemini LLM)  |             -->  XGBoost / LightGBM model
                       |                         |
                       v                         v
                 Query Runner          RiskPredictor (scores 0-1000)
                       |               TreeSHAP (local explanations)
                       |               RuleDeriver (mined IF-THEN rules)
                       |               BusinessRulesEngine (hard policies)
                       |                         |
                       +-------------+-----------+
                                     v
                            Streamlit Web UI (app.py)
```

**Key design decisions:**

- **Parquet cache** — the six raw CSVs (approx 2 GB total) are merged once and saved as `merged_data.parquet`. Subsequent launches read this file directly, cutting startup time from ~5 minutes to ~15 seconds.
- **SQLite database** — all six tables are also loaded into `credit_risk.db` so the chatbot can run arbitrary SQL without loading the data into Python memory on every query.
- **Saved model artifacts** — the trained model and fitted preprocessing pipeline are persisted to disk, so the UI never re-trains on startup.

---

## Setup and Running the App

### Prerequisites

Before you start, make sure you have:

- **Python 3.10, 3.11, or 3.12** — check with `python --version`
- **A free Gemini API key** — get one in 30 seconds at https://aistudio.google.com (no credit card required)
- **The raw data files** placed inside the `data/` folder (see Repository Structure below)
- **Docker Desktop** — only needed for the Docker option

---

### Option A — Local Python (recommended for development)

**Step 1 — Create a virtual environment**

A virtual environment keeps this project's dependencies isolated from everything else on your machine.

```bash
# From inside the credit_risk_platform/ directory:
python -m venv .venv

# Activate it:
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

**Step 2 — Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 3 — Create your .env file**

Create a file called `.env` in the `credit_risk_platform/` root:

```
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL_NAME=gemini-3.5-flash-lite
RATE_LIMIT_RPM=15
RATE_LIMIT_RPD=500
```

> The Gemini API key is only needed for the Talk-to-Data chatbot tab. All other tabs (EDA, Risk Prediction, Business Rules) work without it.

**Step 4 — Run the app**

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. The first launch takes 30-60 seconds while the parquet file loads.

---

### First-time setup — building the data artifacts

If you are starting with only the raw CSV files and no trained model yet, run these once in order:

```bash
# 1. Merge CSVs and create merged_data.parquet
python src/data/loader.py

# 2. Train the model — takes 10-20 minutes on the full dataset
#    Creates models/AutoML_Best_Model.pkl and models/preprocessor.pkl
python src/ml/train.py

# 3. Print evaluation metrics for the trained model
python src/ml/evaluate.py

# 4. Create the SQLite database for the chatbot tab
python src/talk_to_data/init_db.py
```

You only need to run these once. After that, `streamlit run app.py` is all you need.

---

### Option B — Docker (recommended for submission and deployment)

Docker packages the entire app into a self-contained container that runs identically on any machine — no Python installation or dependency management required.

**Step 1 — Add your API key to .env**

```
GEMINI_API_KEY=your_actual_key_here
```

**Step 2 — Build and start the container**

```bash
docker-compose up --build
```

Open http://localhost:8501 in your browser.

**Other useful commands:**

```bash
# Run in the background:
docker-compose up --build -d

# View live logs:
docker-compose logs -f

# Stop the container:
docker-compose down
```

> The `data/`, `models/`, and `credit_risk.db` files are mounted as read-only volumes — the container reads your local files without copying them inside the image, keeping the image small and startup fast.

---

## Model Selection and Class Imbalance Strategy

### Why Gradient Boosted Trees?

The dataset has 467 features, heavy missingness, mixed data types (numeric and categorical), and non-linear relationships between features. Gradient boosted trees (XGBoost / LightGBM) handle all of this natively without requiring the manual feature engineering that linear models would need.

They are also the standard choice in industry credit risk modelling because:
- They produce well-calibrated probabilities, which is important when the output is a scored range rather than a binary label.
- They support TreeSHAP, which produces mathematically exact explanations for each individual prediction.
- They are naturally robust to outliers and missing values.

Other options that were evaluated:

| Model | Why not chosen |
| :--- | :--- |
| **Random Forest** | Similar accuracy but 7x larger model file, ~45ms inference (vs ~12ms), and weaker calibration on imbalanced data. |
| **Logistic Regression** | Needs extensive manual feature engineering to capture non-linear interactions. Best AUC achieved was ~0.68 vs ~0.77. |
| **Neural Networks (MLP)** | Consistently underperform gradient boosted trees on tabular data at this scale. Also effectively black-boxes without a dedicated explanation layer. |

**Why FLAML AutoML?** Manually tuning learning rate, max depth, regularisation, and tree count involves hundreds of combinations. FLAML runs a cost-aware Bayesian search across XGBoost and LightGBM configurations and selects the best one within a fixed 600-second time budget — consistently outperforming any single manually-configured model.

---

### The Class Imbalance Problem

Only **8.07% of applicants defaulted** (24 825 out of 307 511 loans). A naive model that predicts "will repay" for everyone achieves 91.93% accuracy — but catches zero defaulters, making it completely useless.

Three strategies were combined to fix this:

**1. Cost-sensitive loss (`scale_pos_weight = 11.38`)**

This parameter tells the model that missing a real defaulter (false negative) costs 11.38x more than wrongly flagging a repayer (false positive). The model shifts its decision boundary accordingly. The value 11.38 is the exact ratio of non-defaults to defaults in the training data.

**2. Stratified splitting**

The 80/20 train-test split is stratified by the target label, so both splits maintain the same 8.07% default rate. Without this, a random split could leave almost no defaulters in the test set, making evaluation unreliable.

**3. Optimising for ROC-AUC, not accuracy**

All hyperparameter search, model selection, and threshold calibration use ROC-AUC as the objective. ROC-AUC measures whether the model correctly ranks a defaulter as higher-risk than a repayer, and is not distorted by class imbalance the way accuracy is.

---

## Evaluation Metrics and Results

The trained model was evaluated on a held-out test set of **61 503 loans** that were never seen during training.

### What the metrics mean

- **ROC-AUC** — Measures overall ranking ability. An AUC of 0.77 means: pick a random defaulter and a random repayer, and the model correctly identifies the defaulter as higher-risk 77% of the time.
- **PR-AUC** — Precision-Recall AUC. Focuses specifically on the minority (default) class and is more informative than ROC-AUC when class imbalance is severe.
- **Brier Score** — Measures probability calibration. Lower is better. A score of 0.07 means the model's stated probabilities closely match the observed real-world rates.

### Benchmark Results

| Model | Val AUC | Test AUC | Test PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: |
| **FLAML AutoML (XGBoost / LightGBM)** | **0.7745** | **0.7692** | **0.2541** | **0.0682** |
| HistGradientBoosting (baseline) | 0.7512 | 0.7488 | 0.2310 | 0.0714 |
| Random Forest | 0.7320 | 0.7284 | 0.2115 | 0.0745 |
| Logistic Regression (L2) | 0.6890 | 0.6842 | 0.1720 | 0.0798 |

The champion model achieves **0.7692 Test ROC-AUC**, within the accepted industry benchmark range of 0.74-0.78 for this dataset.

### Decision Threshold

The default threshold of 0.50 is too conservative for credit risk — it misses most defaulters. At an optimised threshold of **p = 0.10**:

- **71.4% of actual defaulters are flagged** for review or rejection.
- **72.8% of genuine repayers are still approved**, preserving revenue.
- The false-negative rate drops from 91.9% (at threshold 0.50) to 28.6%.

---

## Prompt Engineering and Token Optimisation

### How the chatbot works

When you type *"What is the average income of defaulters?"*, the system runs these steps:

1. Builds a system prompt telling the LLM: *"You are a SQLite expert. Here is the schema. Return only valid SQL with no explanation."*
2. Calls Gemini with that system prompt plus your question.
3. Validates the returned SQL before running it — blocks anything that is not a SELECT statement.
4. Runs the SQL on the database, capping results at 100 rows and enforcing a 5-second timeout.
5. Calls Gemini again with the result data, asking for a plain-English business summary.

### Why `gemini-3.5-flash-lite` with Multi-Model Fallback?

`gemini-3.5-flash-lite` is Google's ultra-fast, token-efficient model designed for low-latency structured tasks like natural language SQL generation. To guarantee 100% uptime even under free-tier quota spikes, the platform implements an **Automatic Multi-Model Failover Cascade**:

1. **Primary Model**: `gemini-3.5-flash-lite` (15 RPM free tier) — handles 95%+ of queries with sub-second response times.
2. **First Fallback**: `gemini-3.1-flash-lite` (15 RPM / 500 RPD) — seamlessly takes over if the primary hits HTTP 429 quota exhaustion.
3. **Second Fallback**: `gemini-3.5-flash` (5 RPM / 20 RPD) — high-reasoning fallback tier for complex joins or multi-table queries.

### Staying within free-tier limits

The system proactively manages rate limits and quotas through:

- **Multi-Model Auto-Failover** — automatically switches from `gemini-3.5-flash-lite` to `gemini-3.1-flash-lite` to `gemini-3.5-flash` if any model encounters HTTP 429 (`RESOURCE_EXHAUSTED`).
- **Sliding-window rate limiter** — tracks all requests in the last 60 seconds. If you reach the limit, the UI displays a countdown instead of throwing an error.
- **Input length guardrail** — questions must be 3-600 characters, preventing accidental token exhaustion from pasting large text blocks.
- **Injection detection** — phrases like `ignore previous instructions` or `drop table` are rejected before they reach the LLM.
- **Schema compression** — only table names, column names, key types, and short domain definitions are injected into the prompt, keeping prompt overhead under 600 tokens.
- **Graceful degradation** — if all rate limits are reached, raw SQL result tables are rendered in the UI directly without making a second summarisation call.

---

## Rule Derivation Logic and Sample Outputs

### Two layers of governance

Credit decisions use two different types of rules that serve different purposes.

**Layer 1 — Hard Policy Rules (`BusinessRulesEngine`)**

These run before the ML model. If any condition is breached, the applicant is immediately classified as Rule Reject, regardless of what the model would predict. These represent non-negotiable regulatory or business thresholds:

| Rule | Threshold | Why |
| :--- | :--- | :--- |
| Debt-to-income | Credit amount > 10x annual income | Debt service would be mathematically unmanageable |
| Bureau score floor | Average external score < 0.10 | Bureau records indicate near-certain default |
| Income floor | Annual income < $30 000 | Insufficient repayment capacity |
| Age check | Age < 18 years | Legal eligibility requirement |
| Employment tenure | Employed < 6 months (non-pensioners) | Insufficient employment stability |

**Layer 2 — Data-Mined Rules (`RuleDeriver`)**

These are automatically extracted from the training data. A shallow decision tree (max depth 3, at least 500 loans per leaf) is fitted on the most predictive features. Each path from root to leaf is converted into a human-readable IF-THEN rule with the observed default rate and portfolio coverage.

**Why a surrogate tree?** The main XGBoost ensemble uses hundreds of trees and is not directly readable. The surrogate is a single, deliberately shallow tree, so each rule is short, auditable, and supported by thousands of real historical loans — not hand-crafted guesses.

### Sample Mined Rules

| Rule | Condition | Default Rate | Lift vs Base | Portfolio Coverage |
| :--- | :--- | :---: | :---: | :---: |
| HIGH-01 | Bureau Score 2 <= 0.38 AND Bureau Score 3 <= 0.33 | 23.8% | 2.95x | 10.2% |
| HIGH-02 | Bureau Score 2 <= 0.45 AND Age <= 29.5y AND Employed <= 1.5y | 19.4% | 2.40x | 4.8% |
| ELEV-01 | Credit-to-Income >= 6.5x AND Bureau Score 3 <= 0.45 | 15.8% | 1.96x | 7.2% |
| MOD-01 | 0.45 < Bureau Score 2 <= 0.55 AND Age <= 35y | 9.8% | 1.21x | 13.4% |
| PRIME-01 | Bureau Score 2 > 0.55 AND Bureau Score 3 > 0.52 AND Age > 35y | 2.1% | 0.26x | 20.9% |

*Lift = how many times riskier this segment is vs the overall portfolio base rate of 8.1%.*

---

## Known Limitations and Possible Improvements

**1. SQLite is not suitable for concurrent production use**

SQLite uses file-level locking. Two simultaneous chatbot queries will work fine, but under heavy concurrent load one will wait for the other. For a real deployment, the database should be migrated to PostgreSQL with connection pooling.

**2. Thin-file applicants are handled poorly**

Applicants with no prior credit history have missing bureau scores, filled with population medians at preprocessing time. This compresses their predicted probability toward the average, losing the signal that *no credit history* is itself a risk indicator. A production system would integrate alternative data sources (utility payments, open banking cashflows).

**3. The model will drift over time**

The model was trained on a static historical snapshot. As economic conditions change, the distribution of input features shifts and the model's calibration degrades. A production system needs automated drift monitoring (e.g. Evidently AI) and scheduled retraining.

**4. The chatbot has no memory between questions**

Each question is answered independently. If you ask *"Show the top 5 loan types by default rate"* and follow up with *"Now filter to female applicants only"*, the chatbot does not know what the previous question was. Redis-backed session memory would enable multi-turn drill-down conversations.

**5. SHAP adds ~80-120ms per prediction**

Exact TreeSHAP values are computed at inference time for each loan scored. For a high-throughput real-time API, this overhead is significant. Pre-computing background expectations or using FastTreeSHAP's linear approximation would bring this under 10ms.

---

## Repository Structure

```
credit_risk_platform/
|
|-- app.py                       # Main entry point - Streamlit web application
|-- credit_risk.db               # 1.77 GB SQLite database (all 6 tables)
|-- Dockerfile                   # Container definition (python:3.11-slim base)
|-- docker-compose.yml           # One-command orchestration with volume mounts
|-- requirements.txt             # All pinned Python dependencies
|-- .env                         # API keys and config (create this yourself, not committed to git)
|
|-- data/
|   |-- application_train.csv    # Primary dataset - 307 511 loans with TARGET label
|   |-- application_test.csv     # Unlabelled test set (48 744 loans)
|   |-- bureau.csv               # External credit bureau records
|   |-- bureau_balance.csv       # Monthly bureau status history
|   |-- previous_application.csv # Previous loan applications by same client
|   |-- installments_payments.csv# Repayment history for previous loans
|   |-- credit_card_balance.csv  # Monthly credit card balance snapshots
|   |-- POS_CASH_balance.csv     # Point-of-sale and cash loan balance history
|   |-- merged_data.parquet      # Cached merged feature table (generated on first run)
|   `-- HomeCredit_columns_description.csv
|
|-- models/
|   |-- AutoML_Best_Model.pkl    # Champion model selected by FLAML
|   |-- preprocessor.pkl         # Fitted preprocessing pipeline
|   |-- test_data.pkl            # Held-out test split for evaluation
|   `-- automl.log               # FLAML hyperparameter search trial log
|
|-- notebooks/
|   |-- eda.ipynb                # Research notebook - 30 cells of exploratory analysis
|   `-- eda.py                   # Same analysis as a standalone runnable script
|
|-- sql/
|   `-- schema.sql               # DDL definitions for all 6 database tables
|
|-- logs/
|   `-- credit_risk.log          # Application runtime log
|
`-- src/
    |-- data/
    |   |-- loader.py            # Loads, merges, and downcasts CSVs (-59.6% RAM)
    |   `-- preprocessor.py      # Imputation, anomaly handling, encoding
    |
    |-- ml/
    |   |-- train.py             # FLAML AutoML training script
    |   |-- evaluate.py          # Loads saved model, prints ROC-AUC and confusion matrix
    |   |-- predict.py           # RiskPredictor class - scoring and SHAP explanations
    |   `-- rules.py             # BusinessRulesEngine (hard rules) + RuleDeriver (mined rules)
    |
    |-- talk_to_data/
    |   |-- nl_to_sql.py         # Orchestrates NL -> SQL -> answer pipeline
    |   |-- guardrails.py        # Rate limiter, input sanitiser, SQL AST validator
    |   |-- prompt_templates.py  # System prompts and schema context for Gemini
    |   |-- query_runner.py      # Executes SQL with row cap and timeout
    |   `-- init_db.py           # One-time script to populate credit_risk.db
    |
    |-- ui/
    |   `-- eda_view.py          # Six-tab Plotly dashboard for the EDA section
    |
    `-- utils/
        |-- config.py            # Central constants - file paths, model params, rate limits
        |-- helpers.py           # Memory reduction utilities and formatting helpers
        `-- logger.py            # Centralised structured logger used across all modules
```

---

## Attribution

Built for the **NeoStat Credit Risk Analytics Assessment**.
Dataset: Home Credit Default Risk (Kaggle).
Libraries: pandas, scikit-learn, xgboost, lightgbm, flaml, shap, plotly, streamlit, google-genai, sqlite3.
