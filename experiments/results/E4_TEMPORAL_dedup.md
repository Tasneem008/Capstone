# Temporal validation (dedup)

Forward and leave-one-phase-out models drop Political_Phase, Phase_Number, Year, and Days_Since_First_Intake. Those columns identify the window. The random-split model keeps them, which is the comparison H3 asks for.

                  protocol                         held_out  Accuracy  Precision  Recall     F1  ROC_AUC  n_train  n_test
 random_split_all_features                 20% mixed phases    0.8636     0.8557  0.8885 0.8718   0.9509     4398    1100
          forward_test_TP2 Transitional phase Last 5 months    0.7575     0.8228  0.5415 0.6532   0.8209     3600    1200
forward_test_Newly_Elected                    Newly Elected    0.8968     0.9841  0.8853 0.9321   0.9734     4800     698
       leave_one_phase_out                           Stable    0.5133     0.7638  0.3425 0.4729   0.6039     4298    1200
       leave_one_phase_out                         Unstable    0.7492     0.9981  0.6337 0.7752   0.9103     4298    1200
       leave_one_phase_out  Transitional phase 1st 5 months    0.5933     0.3110  0.9776 0.4719   0.8499     4298    1200
       leave_one_phase_out Transitional phase Last 5 months    0.7550     0.7944  0.5652 0.6605   0.8128     4298    1200
       leave_one_phase_out                    Newly Elected    0.8968     0.9841  0.8853 0.9321   0.9734     4800     698

## H3
Random-split F1 is 0.872. Training on earlier windows and testing on Transitional phase 2 gives F1 0.653 (change -0.219). The same protocol on Newly Elected gives F1 0.932 (change +0.060).
This file already drops later copies of an earlier fingerprint. The Newly Elected forward test is then 698 rows that did not match an earlier record, and the F1 does not fall. Identical rows sitting in both the training window and that test window are not what holds the score up. The Transitional phase 2 drop is unchanged, and that drop is the H3 result. Leave-one-phase-out on Unstable is the comparison that does move: on the full file the Newly Elected copies sit in the training side of that split.