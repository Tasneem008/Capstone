# Reduced feature set (E8)

Dropped duplicate encodings: Budget, Budget_Midpoint, Budget_Level_Num, Budget_Academic_Score, Researched_Bin, Phase_Number, Academic_Level_Num, Result_Band, GPA_Academic_Match.

Kept: Result, Course, Country, Institution, Researched, Political_Phase, Budget_Level, Academic_Level, Intake_Month, Year, Season, Days_Since_First_Intake, Researched_Level.

feature_set  n_features  Accuracy  Precision  Recall     F1  ROC_AUC  Brier  cv_f1_mean  cv_f1_low  cv_f1_high
    full_22          22    0.8642     0.8789  0.8578 0.8682   0.9513 0.0904      0.8574     0.8502      0.8647
    reduced          13    0.8633     0.8762  0.8594 0.8677   0.9474 0.0938      0.8560     0.8496      0.8623

Held-out F1 change (reduced minus full): -0.0005. A change inside a few thousandths means the extra encodings were not buying a different decision.