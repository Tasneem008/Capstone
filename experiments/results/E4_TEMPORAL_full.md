# Temporal validation (full)

Forward and leave-one-phase-out models drop Political_Phase, Phase_Number, Year, and Days_Since_First_Intake. Those columns identify the window. The random-split model keeps them, which is the comparison H3 asks for.

                  protocol                         held_out  Accuracy  Precision  Recall     F1  ROC_AUC  n_train  n_test
 random_split_all_features                 20% mixed phases    0.8642     0.8789  0.8578 0.8682   0.9513     4800    1200
          forward_test_TP2 Transitional phase Last 5 months    0.7575     0.8228  0.5415 0.6532   0.8209     3600    1200
forward_test_Newly_Elected                    Newly Elected    0.8867     0.9872  0.8449 0.9105   0.9731     4800    1200
       leave_one_phase_out                           Stable    0.5367     0.7470  0.4131 0.5320   0.6001     4800    1200
       leave_one_phase_out                         Unstable    0.8358     0.9968  0.7619 0.8637   0.9405     4800    1200
       leave_one_phase_out  Transitional phase 1st 5 months    0.5850     0.3072  0.9821 0.4679   0.8472     4800    1200
       leave_one_phase_out Transitional phase Last 5 months    0.7483     0.7757  0.5672 0.6553   0.8216     4800    1200
       leave_one_phase_out                    Newly Elected    0.8867     0.9872  0.8449 0.9105   0.9731     4800    1200

## H3
Random-split F1 is 0.868. Training on earlier windows and testing on Transitional phase 2 gives F1 0.653 (change -0.215). The same protocol on Newly Elected gives F1 0.911 (change +0.042).
H3 is about the forward test, not about every held-out phase scoring worse. The drop on Transitional phase 2 supports H3: a mixed random split overstates what the model does on a later window it has not seen. Newly Elected staying high was checked after later fingerprint copies were removed (`E4_TEMPORAL_dedup.md`). That forward F1 stays at 0.932 on the 698 non-matching rows, so identical-row leakage is not the explanation. The Transitional phase 2 drop, F1 0.653, is the H3 result on both files.