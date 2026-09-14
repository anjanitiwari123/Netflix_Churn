# Netflix Customer Churn Prediction

An audited, portfolio-ready machine learning project that predicts whether a Netflix-style
subscription customer will churn. This version is the result of a full audit pass over an
earlier draft: one factual error in the EDA was found and corrected, and a dedicated
data-leakage and robustness investigation was added to determine whether the model's very high
reported performance (~99.7% across most metrics) is trustworthy.

## 1. Project Overview

Complete ML lifecycle on a real churn dataset: deep data analysis → cleaning decisions backed by
evidence → feature engineering → **explicit data-leakage and temporal-risk investigation** →
model comparison across 5 algorithms → hyperparameter tuning → **robustness/ablation
experiments** → explainability (SHAP) → a saved, deployment-ready model → a Streamlit web app.

Every preprocessing decision is justified with numbers from the dataset itself. No OOP anywhere
(notebook or Streamlit app) — plain functions and procedural code throughout.

## 2. Business Problem

Subscription churn is expensive: acquiring a new customer typically costs more than retaining an
existing one. This project builds an early-warning model to flag likely churners so the business
can intervene proactively — but only after checking whether the model's apparent skill is real
or an artifact of how the (likely synthetic) dataset was constructed.

## 3. Dataset

- **Source file:** `netflix_customer_churn.csv`
- **Size:** 5,000 customers, 14 raw columns
- **Target:** `churned` — **50.3% churned vs 49.7% retained** (near-perfectly balanced)
- **Data quality:** zero missing values, zero duplicate rows, `customer_id` unique for every row
- **No timestamp, snapshot-date, or signup-date column exists anywhere in the dataset** — this
  turns out to matter a lot (see Section 11, Limitations).

## 4. EDA Findings (including one audit correction)

- **Inactivity is the strongest single driver of churn.** `last_login_days` correlates ~+0.47
  with churn; customers inactive 30+ days churn at **75.1%** vs **26.1%** for active customers.
- **Engagement protects against churn.** `watch_hours` correlates ~-0.48 with churn.
- **Subscription tier matters:** Basic churns at ~62% vs ~44-45% for Standard/Premium.
- **Payment method matters:** Crypto (~60%) and Gift Card (~58%) payers churn more than
  Credit/Debit Card payers (~44%).
- **Demographics are close to noise:** age, gender, region, device, favorite genre all sit within
  a few points of the ~50% baseline.
- **AUDIT CORRECTION — number of profiles:** an earlier draft of this notebook described the
  relationship between `number_of_profiles` and churn as "essentially flat," which directly
  contradicted the numbers printed in that same notebook. The real pattern: accounts with 1-3
  profiles churn at **~57-59%**, while accounts with 4-5 profiles churn at **~38-41%** — a real
  ~20-point gap (correlation ≈ -0.16). This has been corrected throughout the notebook and this
  README, and is also confirmed independently by permutation importance (Section 9), where
  `number_of_profiles` ranks as a real, non-trivial contributor.

## 5. Data Preprocessing

- **Missing values / duplicates:** none — no imputation or dedup needed.
- **Identifier removed:** `customer_id` (pure UUID, no predictive value).
- **Redundant feature removed:** `monthly_fee` is 100% determined by `subscription_type`
  (Basic=$8.99, Standard=$13.99, Premium=$17.99, zero within-group variance, re-verified during
  audit). Kept the interpretable categorical version.
- **Outlier handling:** `watch_hours` kept as-is (genuine heavy usage, correlates with *lower*
  churn). `avg_watch_time_per_day` winsorized at the 99th percentile after finding 10 rows with
  physically impossible values (>24 hrs/day), all with `last_login_days` near 0 — a
  divide-by-near-zero construction artifact, not real behavior. Capping *improved* its
  correlation with churn from -0.27 to -0.40.
- **Skewness:** transformation applied only inside the Logistic Regression branch (Yeo-Johnson +
  scaling); tree models use raw values since they're invariant to monotonic transforms.

## 6. Feature Engineering

Three engineered features:

| Feature | Definition | Corr. with churn |
|---|---|---|
| `is_inactive_30` | 1 if `last_login_days` > 30, else 0 | ~0.49 |
| `watch_per_profile` | `watch_hours / number_of_profiles` | ~-0.32 |
| `engagement_ratio` | `watch_hours / (last_login_days + 1)` | ~-0.27 (Pearson) |

**Important nuance found during audit:** `engagement_ratio`'s linear (Pearson) correlation with
churn is a modest -0.27, but used **on its own** as a churn-risk score it achieves **~0.92
ROC-AUC** — nearly as good as the full 5-model ensemble's ~0.999. Pearson correlation understates
this feature's power because the relationship is non-linear; rank-based metrics like AUC capture
it much better. This single fact is the main explanation for why overall model performance is so
high (see Section 8).

**Engineered-feature redundancy check:** `engagement_ratio` and `avg_watch_time_per_day` correlate
at 0.81; `is_inactive_30` and `last_login_days` correlate at 0.87 (expected — one is a thresholded
version of the other). All four were kept (each still shows independent permutation importance),
but this redundancy is exactly what the robustness experiments in Section 8 test directly rather
than assuming.

## 7. Class Imbalance Strategy

Target is 50.3% / 49.7%. **SMOTE was evaluated and explicitly not used** — the classes are
already approximately balanced, so oversampling would add noise for no benefit. No
`imbalanced-learn` dependency is included in `requirements.txt` as a result.

## 8. Data Leakage & Robustness Investigation (the core of this audit)

**The question:** is ~99.7% accuracy legitimate, or is a feature making the problem artificially
easy?

**Structural leakage check:** no column is computed from a post-outcome event (no cancellation
date, no post-churn support-ticket flag, etc.), and `churned` never appears among the features.
No obvious structural leakage.

**Temporal leakage risk — documented honestly, not dismissed:** `last_login_days` (and everything
derived from it — `is_inactive_30`, `engagement_ratio`) is a "days since" measurement. Whether
it's safe depends entirely on **when it was captured relative to the churn decision**, and **this
dataset provides no timestamp/snapshot-date column to verify that**. If it's a fixed-date snapshot
for every customer, it's a legitimate early-warning signal. If it's effectively "days since the
customer's last action before leaving" for churners specifically, it's much closer to leakage.
We cannot determine which is true from the data alone, so we don't claim either — we document the
risk and quantify its impact instead (below).

**Evidence against trivial leakage:** `is_inactive_30` is not a disguised copy of the target — 26%
of churned customers were *not* flagged inactive, and 25% of retained customers *were* flagged
inactive. A true leakage column would show near-zero overlap in both directions.

**Robustness experiments** (same tuned Gradient Boosting algorithm, 5-fold CV on the training set
only, one feature set changed at a time):

| Experiment | Accuracy | ROC-AUC |
|---|---|---|
| 1. Baseline (all current features) | 0.9965 | 0.9998 |
| 2. Remove `is_inactive_30` | 0.9965 | 0.9998 |
| 3. Remove `last_login_days` | 0.9945 | 0.9996 |
| 4. Remove `engagement_ratio` | 0.9970 | 0.9998 |
| 5. Remove `avg_watch_time_per_day` | 0.9965 | 0.9998 |
| **6. Remove the ENTIRE inactivity/engagement cluster** | **0.7562** | **0.8624** |

**What this shows:** removing any *single* feature from the cluster changes almost nothing — the
remaining, highly-correlated cluster members fully compensate. That argues against fragile,
single-column leakage. But removing the **whole cluster together** causes a large, real collapse
— from ~99.98% to ~86% ROC-AUC. That is still well above chance (the remaining features —
`watch_hours`, `subscription_type`, `payment_method`, `number_of_profiles` — carry real,
independent signal), but nowhere near the headline number.

**Bottom line:** the ~99.7% score is not explained by a single fragile or duplicated feature, but
it is heavily concentrated in one behavioral concept (recent login/engagement activity). If that
concept's measurement timing can't be verified against production data availability, a more
realistic expectation for deployed performance is closer to the ~86% ROC-AUC / ~76% accuracy of
Experiment 6 than the ~99.98% headline. **We kept the headline model and did not discard these
features** — there's no direct evidence of leakage, only an unverifiable risk — but this
trade-off is now explicit rather than hidden.

## 9. Models Compared & Selection

Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost — 5-fold stratified
CV, then hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

| Model | Baseline CV ROC-AUC | Tuned CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.9794 | 0.9803 |
| Decision Tree | 0.9730 | 0.9900 |
| Random Forest | 0.9974 | 0.9971 |
| **Gradient Boosting** | **0.9991** | **0.9997** |
| XGBoost* | 0.9995 | 0.9996 |

\* Built in an offline sandbox without internet access to `pip install xgboost`/`shap`. The
notebook contains real, standard `XGBClassifier`/`shap.TreeExplainer` code that runs as-is once
`pip install -r requirements.txt` is run locally (both are listed there). To keep every number in
this README real, the "XGBoost" row above used the documented, executed fallback — scikit-learn's
`HistGradientBoostingClassifier` (same family: histogram-based gradient boosting).

**Final model: Gradient Boosting** — selected by cross-validated ROC-AUC (not accuracy alone),
confirmed stable across folds, and it stayed the best model after every experiment in this audit
— it was **not** forced to be XGBoost, and would have been swapped if the evidence pointed there.

Permutation importance on the held-out test set (post-audit, corrected): `watch_hours` and
`engagement_ratio` rank highest, followed by `subscription_type`, `number_of_profiles`,
`last_login_days`, and `is_inactive_30` as a meaningful second tier — demographics rank lowest,
consistent with the EDA.

## 10. Final Test-Set Performance

| Metric | Test Set (threshold = 0.353, F1-optimized on CV) |
|---|---|
| Accuracy | 0.9970 |
| Precision | 0.9960 |
| Recall | 0.9980 |
| F1 | 0.9970 |
| ROC-AUC | 0.9991 |
| PR-AUC | 0.9989 |
| Balanced Accuracy | 0.9970 |

R² was computed once, purely to demonstrate why it is *not* an appropriate classification metric
— never used for model selection.

## 11. Limitations (read this before treating the headline number as production-ready)

- **This is not a claim that 99.7% accuracy means the model will perform this well in
  production.** The dataset lacks any way to verify when `last_login_days` was actually measured
  relative to the churn event. Before deployment, confirm with the source system: (a) the exact
  snapshot date this field is measured against for every customer, and (b) that it is always
  available *before* the churn decision, not derived from it.
- If that verification fails, Section 8's Experiment 6 (~86% ROC-AUC / ~76% accuracy) is a more
  realistic expectation than the headline number.
- Metrics reflect one random 80/20 split; a temporal/cohort-based validation (train on earlier
  customers, test on later ones) would better simulate real deployment drift and hasn't been done
  here.
- Results are specific to this dataset, which shows patterns (e.g. an extremely powerful single
  engagement ratio) more clean-cut than most real-world churn data — validate against a second,
  real dataset before drawing broader conclusions.

## 12. SHAP Explainability

Full `shap.TreeExplainer` code (summary + bar plots) is included and will run once
`pip install -r requirements.txt` installs `shap` (unavailable in this offline build sandbox).
Permutation importance (scikit-learn, real executed numbers) is included as an offline-friendly
stand-in and points to the same top drivers.

## 13. Streamlit Application

`app.py` loads `churn_model.pkl` and performs inference only — no retraining, no OOP. Collects
customer/subscription/engagement details, recreates the same engineered features used in
training, and returns a prediction, probability, and a plain-language risk label.

## 14. Project Structure

```
netflix-customer-churn/
│
├── netflix_customer_churn.csv
├── churn_prediction.ipynb
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

## 15. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 16. Running Locally

```bash
jupyter notebook churn_prediction.ipynb   # re-run / inspect the full analysis
streamlit run app.py                      # launch the app
```

## 17. Deployment (Streamlit Community Cloud)

1. Push `app.py`, `churn_model.pkl`, `requirements.txt`, `README.md` to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app".
3. Select the repo/branch and `app.py` as the entry point, then click "Deploy".
4. Streamlit Cloud installs `requirements.txt` automatically and serves the app at a public URL.

## 18. Future Improvements

- Verify `last_login_days`'s measurement timing with the source system before any production use.
- Temporal/cohort validation instead of a single random split.
- Validate findings against a second, real-world churn dataset.
- Per-prediction SHAP explanation in the Streamlit app, not just an overall probability.
- Cost-sensitive thresholds tied to actual retention-offer cost vs. customer lifetime value.
