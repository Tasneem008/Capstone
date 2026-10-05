# Methodology and Model Comparison (6,000-Record Dataset)

All scores below are on the **same held-out test set** (20% stratified split, `random_state=42`, n=1,200) unless noted.

## 1. Source methodologies

### Paper 1 — Carballo-Mendívil et al. (2025)

Early-warning dropout prediction using pre-enrollment data.

**Adapted pipeline:**
- 80/20 stratified train/test split
- Random undersampling of majority class **inside training folds only**
- StandardScaler (numeric) + One-Hot Encoding (categorical)
- Models: Logistic Regression, Random Forest, LightGBM, tuned XGBoost
- 5-fold stratified CV; XGBoost hyperparameter grid search (F1 scoring)
- Decision threshold chosen on **out-of-fold training probabilities** (max F1)

**Best Paper 1 result:** XGBoost (Tuned, optimized threshold) @ threshold 0.29
- Accuracy: 0.8517
- Precision: 0.8035
- Recall: 0.9473
- F1: 0.8695
- ROC-AUC: 0.9484
- MCC: 0.7131

### Paper 2 — Niyogisubizo et al. (2022)

Two-layer stacked ensemble for university dropout prediction.

**Adapted pipeline:**
- 80/20 stratified train/test split
- Layer 1: Random Forest, XGBoost, Gradient Boosting (individual pipelines)
- 10-fold CV to produce out-of-fold probabilities for meta-training
- Layer 2: feed-forward neural network (MLP) on Layer-1 OOF predictions
- No undersampling; default threshold 0.50

**Best Paper 2 base learner:** Paper 2 Stacking (RF + XGBoost + GB -> FNN)
- Accuracy: 0.8683 (86.83%)
- Precision: 0.8493
- Recall: 0.9089
- F1: 0.8781
- ROC-AUC: 0.9501
- MCC: 0.7373

Stacking (RF + XGB + GB → FNN) reached F1 0.8065 — slightly below standalone Gradient Boosting (F1 0.8122) on this dataset.

## 2. This project's adopted models (SARP-Net DSS)

| Component | Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Continuation probability + SHAP | Gradient Boosting | 0.50 | 0.8642 | 0.8789 | 0.8578 | 0.8682 | 0.9513 |
| Risk tiers (Low / Medium / High) | XGBoost (tuned) | 0.29 | 0.8517 | 0.8035 | 0.9473 | 0.8695 | 0.9484 |

**Why hybrid?** Paper 2 GB gives the highest F1 on the test set and powers counselor-facing probabilities and SHAP explanations. Paper 1's optimized XGBoost threshold (0.73) provides calibrated risk bands for triage.

## 3. Head-to-head (best model per methodology)

| Methodology | Best model | Thr. | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper 1 | XGBoost (Tuned, optimized threshold) | 0.29 | 0.8517 | 0.8035 | 0.9473 | 0.8695 | 0.9484 | 0.7131 |
| Paper 2 | Paper 2 Stacking (RF + XGBoost + GB -> FNN) | 0.50 | 0.8683 | 0.8493 | 0.9089 | 0.8781 | 0.9501 | 0.7373 |
| Legacy pipeline | Logistic Regression | 0.50 | 0.8950 | 0.8655 | 0.6867 | 0.7658 | 0.8667 | — |
| This project | XGBoost tuned (Paper 1 pipeline) | 0.29 | 0.8517 | 0.8035 | 0.9473 | 0.8695 | 0.9484 | 0.7131 |
| This project | Gradient Boosting (Paper 2 pipeline) | 0.50 | 0.8642 | 0.8789 | 0.8578 | 0.8682 | 0.9513 | 0.7283 |

## 4. Full model leaderboard (all runs)

See `final_methodology_comparison.csv` for every model evaluated.

## 5. Confusion matrix (project primary model)

Gradient Boosting @ 0.50: TN=500, FP=74, FN=89, TP=537

XGBoost @ 0.73: TN=429, FP=145, FN=33, TP=593

## 6. Feature contract (shared across Paper 1 & 2 adaptations)

- 21 engineered features from `master_6000_engineered.csv`
- Target: `Continuation_Bin` (25% positive class)
- Preprocessing: StandardScaler + OneHotEncoder in sklearn pipelines
