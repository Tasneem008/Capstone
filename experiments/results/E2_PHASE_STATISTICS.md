# Phase statistics (E2)

## H1. Continuation x political phase
Continuation_Bin                    0    1
Political_Phase                           
Stable                            435  765
Unstable                          381  819
Transitional phase 1st 5 months   977  223
Transitional phase Last 5 months  694  506
Newly Elected                     381  819

Chi-square = 903.84, df = 4, p = 2.452e-194, Cramer's V = 0.388.
H1 is supported if p < 0.05. V is the effect size (small 0.1, medium 0.3, large 0.5).

## H2. Due diligence inside each phase
                           phase  spearman_rho   p     No  Yes Low  Yes Medium  Yes High
                          Stable        0.2486 0.0 0.5288   0.5706      0.6923    0.8547
                        Unstable        0.6966 0.0 0.2620   0.9550      0.9920    0.9960
 Transitional phase 1st 5 months        0.4998 0.0 0.0026   0.0549      0.1963    0.5326
Transitional phase Last 5 months        0.6760 0.0 0.0425   0.2359      0.5704    0.8627
                   Newly Elected        0.7483 0.0 0.1820   0.9780      0.9891    0.9947

H2 is supported in a phase when rho > 0 and p < 0.05, i.e. higher due diligence ranks with higher continuation.

## Logistic model, Stable as the phase reference
Controls: GPA (z-scored), due diligence level (0-3), academic level (0-2), budget level (0-2). Date features are excluded so the phase term is not just a second copy of the calendar. Intervals are bootstrap 95% percentiles (400 resamples).

                            term   coef  odds_ratio  or_low  or_high
                           GPA_z -0.034       0.966   0.864    1.071
             Due_Diligence_Level  1.525       4.596   4.254    5.059
                  Academic_Level  1.017       2.766   2.042    3.786
                    Budget_Level  0.540       1.716   1.562    1.871
                   Newly Elected  0.095       1.099   0.887    1.378
 Transitional phase 1st 5 months -3.694       0.025   0.018    0.032
Transitional phase Last 5 months -1.992       0.136   0.104    0.174
                        Unstable  0.353       1.423   1.152    1.753

## Verdicts
H1 is supported: continuation and phase are associated (chi-square p = 2.452e-194, Cramer's V = 0.388).
H2 is supported in every phase.
The rank correlation is weakest in Stable, where continuation is already high at Due Diligence = No. In Unstable and Newly Elected the rate jumps from No to Yes Low and then sits near the ceiling, so the gradient is real but most of it is the first step up from No.
After GPA, due diligence, academic level, and budget, the phase terms whose 95% CI excludes 1 are: Transitional phase 1st 5 months OR 0.025 (CI 0.018-0.032); Transitional phase Last 5 months OR 0.136 (CI 0.104-0.174); Unstable OR 1.423 (CI 1.152-1.753). Newly Elected is the comparison to read carefully: its interval can include 1, which means it is not separable from Stable once those controls are in the model.