# Model selection

Ranking uses 5-fold stratified CV on the training split only (4,800 rows). Accuracy is the selection metric. Test numbers below were not used to pick the winner.

                             model  cv_accuracy  cv_accuracy_std  cv_f1  cv_auc
            GB 200 depth 4 lr 0.05       0.8417           0.0112 0.8496  0.9406
              XGB Paper 2 defaults       0.8404           0.0100 0.8483  0.9399
GB current (100, depth 3, lr 0.10)       0.8402           0.0171 0.8483  0.9406
            GB 300 depth 3 lr 0.05       0.8396           0.0132 0.8481  0.9404
          Stack RF+XGB+GB logistic       0.8392           0.0120 0.8479  0.9403
  XGB undersampled (Paper 1 style)       0.8377           0.0069 0.8428  0.9367
           XGB 400 depth 4 lr 0.05       0.8375           0.0061 0.8450  0.9393
           XGB 300 depth 6 lr 0.05       0.8362           0.0118 0.8442  0.9362
              HistGradientBoosting       0.8346           0.0099 0.8418  0.9349
                      LightGBM 300       0.8344           0.0057 0.8417  0.9343
                 Random Forest 200       0.8344           0.0148 0.8421  0.9313
               Logistic regression       0.8335           0.0153 0.8364  0.8972

## Locked test (threshold 0.50)

                             model  threshold  Accuracy  Precision  Recall     F1  ROC_AUC    MCC
            GB 200 depth 4 lr 0.05        0.5    0.8600     0.8742  0.8546 0.8643   0.9498 0.7200
GB current (100, depth 3, lr 0.10)        0.5    0.8642     0.8789  0.8578 0.8682   0.9513 0.7283
              XGB Paper 2 defaults        0.5    0.8517     0.8636  0.8498 0.8567   0.9485 0.7031
          Stack RF+XGB+GB logistic        0.5    0.8625     0.8773  0.8562 0.8666   0.9509 0.7250

## Decision

CV winner: GB 200 depth 4 lr 0.05 (0.8417).
Adopted project model: GB current (100, depth 3, lr 0.10).
A tuned tree is within 0.3 points of the current Gradient Boosting CV accuracy. The current model stays. The extra fit is not a real gain.

The CV-best Gradient Boosting (200 trees, depth 4) scored 0.860 accuracy on the locked test. The current model scored 0.864, with higher F1 (0.868), AUC (0.951), and MCC (0.728).

The saved Paper 2 feedforward stack still has the highest single-split accuracy, 0.8683. That is about 0.4 points above Gradient Boosting, roughly 5 test rows out of 1,200. The same base models with a logistic stack scored 0.839 CV accuracy, below Gradient Boosting at 0.840. Fold-to-fold accuracy for the current model moves by 1.7 points. The stack's test edge is smaller than that, so it is not a reliable gain. Undersampled XGBoost, which is the Paper 1 setup, is worse on this near-balanced file (CV accuracy 0.838).