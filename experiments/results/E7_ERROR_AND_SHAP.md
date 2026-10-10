# Error analysis and phase-stratified SHAP (E5, E7)

Predictions use the production Gradient Boosting model at the current low-risk cutoff 0.54. A false negative is a student who continued but was scored below that cutoff.

error_type
OK    1034
FN     112
FP      54

## Political_Phase
                                    n  false_negatives  false_positives  fn_rate  fp_rate
Political_Phase                                                                          
Stable                            204               38               36    0.186    0.176
Unstable                          248               23                1    0.093    0.004
Transitional phase 1st 5 months   257               22                6    0.086    0.023
Transitional phase Last 5 months  240               17               11    0.071    0.046
Newly Elected                     251               12                0    0.048    0.000

## Researched
              n  false_negatives  false_positives  fn_rate  fp_rate
Researched                                                         
No          450               74                8    0.164    0.018
Yes Low     206               13                8    0.063    0.039
Yes Medium  261               14               22    0.054    0.084
Yes High    283               11               16    0.039    0.057

## Budget_Level
                n  false_negatives  false_positives  fn_rate  fp_rate
Budget_Level                                                         
Low           420               33               19    0.079    0.045
Medium        365               37               17    0.101    0.047
High          415               42               18    0.101    0.043

Due diligence is stored three ways in the production model (Researched, Researched_Level, Researched_Bin), so its importance is split across those columns. Read them as one factor. Budget is split the same way. E8 drops the extra encodings and the F1 does not move.

## Mean absolute SHAP by phase (top drivers)
                                  Researched  Researched_Level  Budget_Academic_Score  Days_Since_First_Intake  Researched_Bin  Result  Budget  Budget_Level
Political_Phase                                                                                                                                             
Stable                                0.5698            0.6146                 0.2655                   0.5561          0.1601  0.1102  0.0393        0.0583
Unstable                              0.9067            0.7752                 0.4048                   0.1110          0.4338  0.1402  0.0957        0.0889
Transitional phase 1st 5 months       0.7378            0.7380                 0.4293                   0.4088          0.2674  0.1109  0.1631        0.1255
Transitional phase Last 5 months      0.8765            1.0220                 0.6947                   0.3332          0.3218  0.1320  0.0771        0.0645
Newly Elected                         0.9816            0.8771                 0.4340                   0.6336          0.4672  0.1225  0.0919        0.0973