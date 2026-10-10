# Limitations (Capstone C)

These limits are part of the defense, not footnotes to hide.

## Data

- The 6,000 rows come from one agency's applicant records for Bangladeshi students across five political windows. They are not a random sample of all students in the country.
- Shared fingerprints between Unstable and Newly Elected (see `experiments/results/DATA_AUDIT.md`) are not yet explained by the agency. Until they are, the full-file scores and the deduplicated scores (E9) are both reported. The deduplicated run is the conservative one.
- Some names end in a number. The audit counts them and does not print them. The thesis and the app must not show raw names.
- In-file `Political_Phase` is wrong for the Stable file and the Newly Elected file. Modeling uses the file-level period. Say that explicitly so it does not look like a silent edit.
- GPA is tightly bunched. It is a weak standalone separator. That is a data fact, not a modeling failure.

## Timing and leakage

- Due diligence is the strongest predictor in the ablation. That result is only valid for counseling if the agency records the level **before** the continuation outcome. This is not provable from the spreadsheet. It is an open question for the data provider.
- Political phase is a function of intake date. A random split lets the model use the window's base rate. RQ3 (forward validation and leave-one-phase-out, with window columns removed) is the number to quote for a future period. The random-split F1 is an in-period ceiling, not the deployment number. H3 predicts the forward number will be lower. If it is, that supports H3.

## Models

- One agency, one country context, five hand-defined windows. There is no external campus or second agency.
- Stacking can score slightly higher than Gradient Boosting on a single split. The DSS keeps Gradient Boosting because SHAP TreeExplainer applies to it. The repeated-CV test in E3 is the check on whether that gap is noise.
- The counseling cutoff is the E6 value: the lowest calibrated threshold with held-out precision of at least 0.90, currently 0.54. Medium risk from 0.40 up to that cutoff is a counseling band, not a third learned class. The F1-max threshold is recorded and is not used as the low-risk label.
- Class balance changed when the dataset was revised (about 25% continuation before, about 52% after). Older 91% accuracy figures are not comparable. The old majority class was 75% "No". The new majority baseline is about 52%.

## What we do not claim

- We did not beat the published Paper 1 or Paper 2 numbers. We re-ran their training procedures on our data.
- The tool does not decide whether a student continues. A false negative is a student who needed a conversation and might not be flagged. E7 profiles those errors. The counselor stays in the loop.
- The counselor study is a protocol until it is run. Do not describe it as completed.

## Ethics

- Permission from the agency to use the records for this thesis.
- Anonymize names in every appendix, slide, and demo.
- No automated denial of counseling or of an application.
