# Dataset Description Report

## 1. Overview

This dataset collection supports a supervised machine learning study of **international student program continuation** under varying political periods. Each record represents one prospective or enrolled student application profile, linking academic background, destination preferences, financial capacity, research behavior, and the political phase at time of intake.

The analytic target variable is **Continuation** (whether the student continues with the study pathway). The current modeling pipeline (`main.py`, `model_comparison.py`) focuses on the two transitional-phase subsets and compares Logistic Regression and Random Forest classifiers.

---

## 2. Data Corpus

All source files are stored under `Datasets/` in Microsoft Excel (`.xlsx`) format. Each file contains **1,200 rows** and **12 columns**, with a consistent schema across political periods.

| File | Political Phase Label | Intake Period (approx.) | Records | Continuation Rate |
|------|------------------------|-------------------------|---------|-------------------|
| `stable.xlsx` | Stable / Transitional* | 1 Nov 2023 – 31 Mar 2024 | 1,200 | 15.00% |
| `unstable.xlsx` | Unstable | 1 Apr 2024 – 7 Aug 2024 | 1,200 | 31.75% |
| `transitional phase 1.xlsx` | Transitional phase 1st 5 months | 8 Aug 2024 – 30 Jun 2025 | 1,200 | 18.58% |
| `transitional phase 2.xlsx` | Transitional phase Last 5 months | 1 Sep 2025 – 31 Jan 2026 | 1,200 | 42.17% |
| `Transitional.xlsx` | Newly Elected | 1 Feb 2026 – 30 Jun 2026 | 1,200 | 17.50% |

\*The `Political_Phase` field in `stable.xlsx` is labeled `"Transitional"` in-file, while the file name indicates a stable period. Interpretation should follow the project’s period definitions.

**Total corpus size:** 6,000 student records across five political windows.

**Pipeline focus:** Transitional Phase 1 and Phase 2 only (2,400 records after concatenation for modeling).

---

## 3. Variable Dictionary (Raw Schema)

| Variable | Type | Description |
|----------|------|-------------|
| `Name` | Categorical (identifier) | Student name; anonymized in preprocessing to `STU_XXXX` |
| `Result` | Numeric (continuous) | Academic GPA / result score (approx. 2.0–4.0 scale) |
| `Course` | Categorical | Intended program of study (23 unique fields) |
| `Country` | Categorical | Destination country (8 countries) |
| `Institution` | Categorical | Destination institution (≈18–19 unique values) |
| `Date Intake` | Date (day-month-year text) | Intake / application date |
| `Budget` | Ordinal categorical | Self-reported budget band: `Below 10k`, `10-15K`, `16-20K` |
| `Researched` | Binary categorical | Whether the student researched options: `Yes` / `No` |
| `Political_Phase` | Categorical | Political period label for the record |
| `Budget_Level` | Ordinal categorical | Encoded budget tier: `Low`, `Medium`, `High` |
| `Academic_Level` | Ordinal categorical | Academic standing band: `Low`, `Medium`, `High` |
| `Continuation` | Binary categorical (**target**) | `Yes` = continues; `No` = does not continue |

### 3.1 Destination Coverage

**Countries:** Australia, Canada, Germany, Ireland, Malaysia, New Zealand, UK, USA.

**Courses (23):** AI, Accounting, Agriculture, Automotive Engineering, Business Administration, Business Analytics, Civil Engineering, Computer Science, Cyber Security, Data Science, Electrical Engineering, Environmental Science, Finance, Hospitality Management, IT, Law, Marketing, Mechanical Engineering, Nursing, Psychology, Public Health, Software Engineering, Supply Chain Management.

Institutions include universities such as Technical University of Munich, University of Auckland, University College Dublin, Arizona State University, University of Malaya, University of Toronto, and related partners.

---

## 4. Descriptive Profile — Modeling Subsets

### 4.1 Transitional Phase 1 (`transitional phase 1.xlsx`)

| Attribute | Value |
|-----------|--------|
| Sample size | 1,200 |
| Political label | Transitional phase 1st 5 months |
| Mean GPA (`Result`) | 3.21 (SD ≈ 0.31; range 2.03–3.95) |
| Continuation = Yes | 223 (18.58%) |
| Continuation = No | 977 (81.42%) |
| Budget bands | 10–15K: 417; Below 10k: 399; 16–20K: 384 |
| Academic level | Medium-dominant (1,079 Medium; 98 High; 23 Low) |
| Researched = Yes | 606 (50.5%) |

### 4.2 Transitional Phase 2 (`transitional phase 2.xlsx`)

| Attribute | Value |
|-----------|--------|
| Sample size | 1,200 |
| Political label | Transitional phase Last 5 months |
| Mean GPA (`Result`) | 3.20 (SD ≈ 0.33; range 2.02–3.95) |
| Continuation = Yes | 506 (42.17%) |
| Continuation = No | 694 (57.83%) |
| Budget bands | Below 10k: 412; 16–20K: 411; 10–15K: 377 |
| Academic level | Medium-dominant (1,057 Medium; 107 High; 36 Low) |
| Researched = Yes | 624 (52.0%) |

### 4.3 Cross-Phase Observation

GPA distributions are broadly similar between Phase 1 and Phase 2, while **continuation rates differ sharply** (≈18.6% vs ≈42.2%). This supports the research premise that political-period context may relate to continuation outcomes beyond academic preparedness alone.

---

## 5. Target Variable and Class Balance

For the combined transitional modeling set (n = 2,400):

| Class | Approximate Count | Share |
|-------|-------------------|-------|
| Continuation = No | 1,671 | ≈69.6% |
| Continuation = Yes | 729 | ≈30.4% |

The target is moderately imbalanced toward non-continuation. Stratified train/validation/test splitting (as implemented in `model_comparison.py`) is appropriate.

---

## 6. Data Quality Assessment

Across the five source files examined:

| Check | Result |
|-------|--------|
| Missing values | None observed (0 nulls per file) |
| Exact duplicate rows | None in Phase 1 / Phase 2; name reuse occurs in Stable/Unstable (non-unique `Name`) |
| Schema consistency | Same 12 columns across all period files |
| Encoding noise | Budget label casing variants (`Below 10k` vs `10-15K`) are consistent within files and mapped in preprocessing |

**Privacy note:** Raw files contain personal names. The cleaning pipeline replaces `Name` with anonymized IDs (`STU_0001`, …) before exporting modeling CSVs.

---

## 7. Preprocessing and Derived Features

The cleaning script transforms raw Phase 1 / Phase 2 tables into:

- `Datasets/transitional_phase_1_clean.csv`
- `Datasets/transitional_phase_2_clean.csv`

Each cleaned file expands from 12 raw columns to **20 modeling features**, including:

| Derived Feature | Construction |
|-----------------|--------------|
| `Intake_Month`, `Intake_Year`, `Intake_Season` | Extracted from `Date Intake` (season coded 1–4) |
| `Days_Since_First_Intake` | Days from the earliest intake across both phases |
| `Budget_Midpoint` | Midpoint mapping: Below 10k → 8000; 10–15K → 12500; 16–20K → 18000 |
| `Phase_Number` | 1 or 2 from political-phase label |
| Ordinal encodings | `Budget_Level`, `Academic_Level` → High=2, Medium=1, Low=0 |
| Binary encodings | `Researched`, `Continuation` → Yes=1, No=0 |
| Label encodings | `Country`, `Course`, `Political_Phase`, `Institution` |
| `Result_Band` | GPA bands via cut points (0–2.99 / 3.00–3.39 / 3.40–4.0) |
| `Budget_Academic_Score` | Sum of budget and academic ordinal levels |
| `GPA_Academic_Match` | Indicator that `Result_Band` matches `Academic_Level` |

`Date Intake` is dropped after feature extraction. Numeric feature matrices from the cleaned CSVs form the model input; the identifier `Name` is excluded from feature matrices where non-numeric filtering is applied.

---

## 8. Intended Analytical Use

1. **Descriptive analysis** of continuation rates across political periods, countries, and budget tiers.  
2. **Supervised classification** of `Continuation` using academic, financial, institutional, and temporal features.  
3. **Period comparison** (e.g., Kolmogorov–Smirnov tests on GPA / budget-derived variables between Phase 1 and Phase 2).  
4. **Feature importance** analysis (Random Forest) to identify drivers of continuation under transitional conditions.

Supporting analytical outputs produced by the pipeline include `quality_report.txt`, `model_comparison_results.csv`, feature-importance tables, ROC curves, and confusion matrices.

---

## 9. Limitations and Notes

1. **Period files unused in current models:** `stable.xlsx`, `unstable.xlsx`, and `Transitional.xlsx` share the same schema but are not yet ingested by `main.py` / `model_comparison.py`.  
2. **Temporal gap:** No intake records were observed between 1 Jul 2025 and 31 Aug 2025 when comparing Phase 1 and Phase 2 date ranges.  
3. **Label ambiguity:** The stable-period file’s in-file `Political_Phase` value (`"Transitional"`) may require clarification before multi-period pooled analysis.  
4. **Synthetic / curated character of names:** Dense sequential name suffixes and highly structured class balances suggest curated research data; external generalizability should be stated cautiously.  
5. **Class imbalance:** Especially severe in Phase 1 and stable periods; metrics beyond accuracy (precision, recall, F1, ROC-AUC) should be prioritized.

---

## 10. Summary

The dataset is a structured, complete, multi-period panel of **6,000 student intake records** with a shared 12-variable schema describing academics, destination choice, budget, research behavior, political context, and continuation outcome. The active modeling subset comprises **2,400 transitional-phase records**, cleaned and feature-engineered into 20 columns, for binary classification of student continuation under political transition.
