# Defense question bank

Use the one-sentence problem statement in `docs/THESIS_CORE.md` before any metric.

## What exactly are you trying to accomplish?

- **Tests:** whether the thesis has a question or only a model.
- **Answer:** Political instability in 2023-2026 changed whether students follow through on studying abroad. We measure that shift (RQ1), which factors move with it (RQ2), whether a model trained on past periods holds up later (RQ3), and whether counselors can use a calibrated warning (RQ4).
- **Evidence:** `docs/THESIS_CORE.md`, the research design figure, E2, E4, E5, E6.

## Isn't this just training classifiers?

- **Tests:** depth beyond a semester project.
- **Answer:** The classifier is the instrument. The results are the phase effect, the gap between a random split and a future period, the drivers inside each phase, and a counselor protocol for the tool.
- **Evidence:** E2 forest plot, E4 forward-vs-random chart, E5 phase SHAP, `docs/COUNSELOR_EVALUATION_PROTOCOL.md`.

## Is the data real? Why are rows duplicated?

- **Tests:** integrity.
- **Answer:** The records are agency applicant files. The audit found repeated fingerprints, mostly across Unstable and Newly Elected. We did not delete them quietly. Full-data and deduplicated results are both in the thesis. A written note from the agency on why those rows match is still required.
- **Evidence:** `experiments/results/DATA_AUDIT.md`, `experiments/results/E3_BASELINES_dedup.md`, `experiments/results/E4_TEMPORAL_dedup.md`.

## Your features include the date and the phase. Isn't the model just memorizing the period?

- **Tests:** leakage awareness.
- **Answer:** Under a random split, yes, it can use the window base rate. That is why leave-one-phase-out and forward tests drop phase, phase number, year, and days since the first intake. Quote the forward F1, not only the random-split F1.
- **Evidence:** E4.

## Is due diligence known before the outcome?

- **Tests:** target leakage.
- **Answer:** We cannot see the form timestamp in the file. The agency has to confirm the field is filled at intake. If it is filled after the outcome, we drop it and the ablation is the evidence of how much the model depended on it.
- **Evidence:** ablation F1 drop for behavioral features, `docs/LIMITATIONS.md`.

## Why Gradient Boosting when stacking scores higher?

- **Tests:** model selection.
- **Answer:** On the 5x5 CV, Gradient Boosting F1 is about 0.857 versus about 0.840 for logistic regression (Nadeau-Bengio p < 0.001) and about 0.839 for the repeated RF+XGB+GB stack (p about 0.006). That stack is not the Paper 2 feedforward stack. The feedforward stack is higher on the single published split and was not re-run 25 times. Gradient Boosting stays the deployed model because the explanation method fits it exactly. Quote the repeated-CV interval, not the one-split leaderboard.
- **Evidence:** `experiments/results/E3_BASELINES_full.md`.

## Are you better than Paper 1 and Paper 2?

- **Tests:** honesty.
- **Answer:** No. We re-implemented their procedures on our records. Their published scores use other students and other labels.
- **Evidence:** opening paragraph of `results_comparison/METHODOLOGY_COMPARISON.md`.

## Why was accuracy about 91% before and about 86% now?

- **Tests:** whether the story is moving the goalposts.
- **Answer:** The revised files changed the due diligence coding and the continuation balance. The old file was about 75% non-continuation, so a constant "No" already scored 75%. The revised file is about 52% continuation, so the majority baseline is about 52%. 86% on the revised file is a larger lift over that baseline than 91% was over 75%.
- **Evidence:** E1 continuation rate, E3 majority-class row.

## How did you pick the risk cutoff?

- **Tests:** arbitrary thresholds.
- **Answer:** The low-risk cutoff is the lowest calibrated probability that still has at least 90% precision on the held-out test. On the revised file that cutoff is 0.54. Medium risk is 0.40 up to 0.54. The F1-max cutoff (about 0.33) is kept in the operating-point file as a record and is not the counseling threshold. The old 0.73 value belonged to the previous, imbalanced file.
- **Evidence:** `experiments/results/E6_CALIBRATION.md` and `results_paper1/operating_point.json`.

## What if the model is wrong for a student?

- **Tests:** harm.
- **Answer:** False negatives are students who continued but were scored below the cutoff. E7 breaks them out by phase, due diligence, and budget. The tool does not approve or reject anyone. The counselor decides.
- **Evidence:** `experiments/results/E7_ERROR_AND_SHAP.md`.

## Ethics?

- **Answer:** Agency permission, no names in slides or the demo, no automated decision.
- **Evidence:** `docs/LIMITATIONS.md`.

## What did Capstone C change after the panel?

- **Answer:** The topic stayed. The goal was rewritten as four questions. We added a data audit, a statistical phase effect, baselines with uncertainty, temporal validation, calibration, error analysis, a reduced feature check, a duplicate sensitivity run, and a counselor study protocol.
- **Evidence:** `docs/THESIS_CORE.md`.
