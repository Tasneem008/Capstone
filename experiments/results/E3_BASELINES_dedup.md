# Baselines and repeated CV (dedup)

Repeated stratified CV: 5 repeats x 5 folds. Intervals are 95% t intervals on the 25 fold F1 scores. Stacking uses a logistic meta-learner and 2-fold inner CV so the repeated comparison finishes. The single-split FNN stack remains in `results_paper2/paper2_model_comparison.csv`.

                              model   tag  f1_mean  f1_low  f1_high  n_folds
                  Gradient Boosting dedup   0.8489  0.8443   0.8535       25
                            XGBoost dedup   0.8481  0.8442   0.8520       25
                      Random Forest dedup   0.8398  0.8359   0.8438       25
                Logistic regression dedup   0.8271  0.8231   0.8310       25
Stacking (RF+XGB+GB, logistic meta) dedup   0.8183  0.8143   0.8224       25
                Phase-only logistic dedup   0.7176  0.7134   0.7219       25
                     Majority class dedup   0.6861  0.6860   0.6862       25

## Nadeau-Bengio corrected resampled t-test on F1
 mean_diff       t  df      p                             model_a                             model_b metric   tag
    0.0218  3.9524  24 0.0006                   Gradient Boosting                 Logistic regression     F1 dedup
    0.0305  4.2299  24 0.0003                   Gradient Boosting Stacking (RF+XGB+GB, logistic meta)     F1 dedup
   -0.0087 -1.3093  24 0.2028 Stacking (RF+XGB+GB, logistic meta)                 Logistic regression     F1 dedup

## McNemar on the random_state=42 holdout
 a_only_correct  b_only_correct      p                             model_a                             model_b   tag
             67              38 0.0060                   Gradient Boosting                 Logistic regression dedup
              9               8 1.0000                   Gradient Boosting Stacking (RF+XGB+GB, logistic meta) dedup
             71              43 0.0111 Stacking (RF+XGB+GB, logistic meta)                 Logistic regression dedup

Read the majority and phase-only rows as the floor. The jump from phase-only to the full logistic is the student-feature lift on a random split. It does not replace the forward test in E4.
The stacking row here is RF + XGBoost + Gradient Boosting with a logistic meta-learner and 2-fold inner CV. It is not the Paper 2 feedforward stack in `results_paper2/paper2_model_comparison.csv`. That single-split stack is a different procedure and is not inside this repeated test.