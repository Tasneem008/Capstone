# Counselor evaluation protocol (E10)

This is the study design. It is not a completed user study. Run it with counselors from the agency that supplied the records, after the data-timing questions in `docs/LIMITATIONS.md` are answered.

## Purpose

RQ4 asks whether the decision-support system is useful to a counselor, not only whether the model has a high F1. This protocol measures agreement, trust, and whether the SHAP explanation changes the counselor's next step.

## Participants

- 3 to 5 counselors who already advise students on study-abroad continuation.
- They must not be the students in the dataset.
- Stop at 5. This is a usability and agreement study, not a powered trial.

## Materials

- 10 anonymized cases per counselor, drawn from the held-out test split.
- Stratify the 10 cases so each counselor sees at least two Low, two Medium, and two High risk profiles under the current cutoff.
- Each case shows only features a counselor would know at intake: GPA, academic level, budget level, due diligence, course, country, institution, political phase, intake date.
- Do not show the recorded continuation label until after the counselor has answered.
- The system shows continuation probability, risk tier, and the local SHAP waterfall.

Replace `Name` with `Case 01` ... `Case 10` on the sheet. Do not export the raw name column.

## Procedure (about 40 minutes)

1. Explain that the tool is advisory and the counselor's judgment is the decision.
2. For each case, before revealing the model:
   - Ask: "Would you flag this student for follow-up?" (Yes/No)
   - Ask: "What is the main reason?"
3. Reveal the probability, tier, and SHAP plot.
4. Ask: "Would you change the follow-up decision after seeing this?" (Yes/No)
5. Ask the five Likert items below.
6. After all 10 cases, administer the System Usability Scale below.
7. One open question: "What would you not trust this tool to do?"

## Likert items (1 = strongly disagree, 5 = strongly agree)

1. The risk tier matches how I would prioritize this student.
2. The explanation shows factors I can actually discuss with the student.
3. I would use this as a first-pass list, not as the decision.
4. I understand why this student was flagged.
5. I would be comfortable explaining this score to the student.

## System Usability Scale

Brooke (1996). 1 = strongly disagree, 5 = strongly agree. Odd items are positive. Even items are negative.

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Score: for each odd item use (response - 1), for each even item use (5 - response), sum the ten contributions, multiply by 2.5. Range is 0 to 100. The published average is about 68.

## Case sheet

`experiments/e10_case_sheets.py` writes ten anonymized held-out cases to `experiments/results/e10_counselor_cases.md` and the answer key, with continuation hidden from the counselor sheet, to `experiments/results/e10_researcher_key.csv`. Regenerate that sheet if the cutoff changes. Do not paste names into either file.

## Measures

- Agreement: percent of cases where the counselor's pre-tool follow-up flag matches High or Medium risk (flag) vs Low risk (no flag). Report a confusion table, not a single accuracy number.
- Decision change: percent of cases where step 4 is Yes.
- Trust: mean and range of the five Likert items. With n <= 5 counselors, do not report a p-value.
- Usability: SUS score per counselor and the median. A median below 68 is a negative result and must be reported as such.

## What would weaken the claim

- Counselors flag a different set of students from the tool on most cases.
- Median SUS below 68.
- Likert item 2 below 3, which would mean the SHAP plot is not actionable.

Those outcomes stay in the thesis. They limit RQ4. They do not erase RQ1-RQ3.

## Ethics

- Agency permission to show anonymized records to its own counselors.
- No student names on the case sheets.
- Counselors are told the tool does not decide continuation.
- Store responses without counselor names if they ask for that.

## Analysis write-up

One table of agreement, one table of Likert means, the SUS median, and a short quote list from the open question. No model retraining based on this sample.
