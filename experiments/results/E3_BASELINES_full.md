# Baselines and repeated CV (full)

Repeated stratified CV: 5 repeats x 5 folds. Intervals are 95% t intervals on the 25 fold F1 scores. Stacking uses a logistic meta-learner and 2-fold inner CV so the repeated comparison finishes. The single-split FNN stack remains in `results_paper2/paper2_model_comparison.csv`.

                              model  tag  f1_mean  f1_low  f1_high  n_folds
                  Gradient Boosting full   0.8569  0.8536   0.8601       25
                            XGBoost full   0.8538  0.8507   0.8569       25
                      Random Forest full   0.8469  0.8443   0.8495       25
                Logistic regression full   0.8398  0.8370   0.8426       25
Stacking (RF+XGB+GB, logistic meta) full   0.8391  0.8351   0.8430       25
                Phase-only logistic full   0.7139  0.7093   0.7185       25
                     Majority class full   0.6859  0.6858   0.6861       25

## Nadeau-Bengio corrected resampled t-test on F1
 mean_diff       t  df      p                             model_a                             model_b metric  tag
    0.0170  6.2708  24 0.0000                   Gradient Boosting                 Logistic regression     F1 full
    0.0178  3.0273  24 0.0058                   Gradient Boosting Stacking (RF+XGB+GB, logistic meta)     F1 full
   -0.0007 -0.1344  24 0.8942 Stacking (RF+XGB+GB, logistic meta)                 Logistic regression     F1 full

Read the majority and phase-only rows as the floor. The jump from phase-only to the full logistic is the student-feature lift on a random split. It does not replace the forward test in E4.

The stacking row here is RF + XGBoost + Gradient Boosting with a logistic meta-learner and 2-fold inner CV. It is not the Paper 2 feedforward stack in `results_paper2/paper2_model_comparison.csv`. That single-split stack is a different procedure and is not inside this repeated test.

## McNemar on the random_state=42 holdout
 a_only_correct  b_only_correct      p                             model_a                             model_b  tag
             64              37 0.0093                   Gradient Boosting                 Logistic regression full
             18              10 0.1849                   Gradient Boosting Stacking (RF+XGB+GB, logistic meta) full
             61              42 0.0756 Stacking (RF+XGB+GB, logistic meta)                 Logistic regression full