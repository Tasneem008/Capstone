# Data audit (E1)

Rows: 6000. Columns checked: model features plus name, date, and continuation.

## Missing values
No missing values in the engineered table.

## Continuation by phase
                                  continuation_pct     n
Political_Phase                                         
Stable                                       63.75  1200
Unstable                                     68.25  1200
Transitional phase 1st 5 months              18.58  1200
Transitional phase Last 5 months             42.17  1200
Newly Elected                                68.25  1200

Overall continuation: 52.20% (3132 Yes / 2868 No).

## Date windows
- Stable: 2023-11-01 to 2024-03-31 (expected 2023-11-01 to 2024-03-31) [OK]
- Unstable: 2024-04-01 to 2024-08-07 (expected 2024-04-01 to 2024-08-07) [OK]
- Transitional phase 1st 5 months: 2024-08-08 to 2025-06-30 (expected 2024-08-08 to 2025-06-30) [OK]
- Transitional phase Last 5 months: 2025-09-01 to 2026-01-31 (expected 2025-09-01 to 2026-01-31) [OK]
- Newly Elected: 2026-02-15 to 2026-07-15 (expected 2026-02-15 to 2026-07-15) [OK]

## Due diligence
            continuation_pct     n
Researched                        
No                     22.20  2243
Yes High               85.28  1488
Yes Low                52.53  1009
Yes Medium             66.27  1260

## Names
Names ending in a space and a number: 2400 of 6000. Individual names are not listed in this report.
Unique names: 3976.

## In-file Political_Phase labels
Modeling uses the workbook period, not the cell. These are the labels stored in each file before that correction.
- Stable (updated_datasheet/stable_revised.xlsx): Transitional n=1200. DIFFERS from the file period.
- Unstable (updated_datasheet/unstable_revised.xlsx): Unstable n=1200. matches the file period.
- Transitional phase 1st 5 months (updated_datasheet/transitional_phase_1_revised.xlsx): Transitional phase 1st 5 months n=1200. matches the file period.
- Transitional phase Last 5 months (updated_datasheet/transitional_phase_2 revised.xlsx): Transitional phase Last 5 months n=1200. matches the file period.
- Newly Elected (updated_datasheet/newly elected_revised_dates_feb15_jul15_2026.xlsx): Unstable n=1200. DIFFERS from the file period.

## Duplicates
Exact duplicate rows: 0.
Rows sharing a cross-field fingerprint (name, academics, destination, budget, due diligence, continuation): 1004.

                         phase_a                          phase_b  shared_fingerprints
                          Stable                         Unstable                    0
                          Stable  Transitional phase 1st 5 months                    0
                          Stable Transitional phase Last 5 months                    0
                          Stable                    Newly Elected                    0
                        Unstable  Transitional phase 1st 5 months                    0
                        Unstable Transitional phase Last 5 months                    0
                        Unstable                    Newly Elected                  502
 Transitional phase 1st 5 months Transitional phase Last 5 months                    0
 Transitional phase 1st 5 months                    Newly Elected                    0
Transitional phase Last 5 months                    Newly Elected                    0

A looser fingerprint that ignores academic level (name, result, course, country, institution, budget, due diligence, continuation) shares 568 fingerprints on its largest phase pair. E9 deduplicates on the stricter fingerprint, which also requires the academic level to match.

## Dedup rule used later in E9
Keep the earliest row when the fingerprint repeats. Removed 502 later copies and 0 residual copies. Rows after: 5498 (from 6000).

## Outliers on GPA
Result min 2.00, max 3.99, mean 3.158. Values outside 2.0-4.0: 0.

## What this audit does not prove
- It does not prove the records are independent students. Shared fingerprints across Unstable and Newly Elected need a written explanation from the agency.
- It does not prove Due_Diligence was recorded before Continuation. That timing has to come from the agency, not from the spreadsheet.
- Modeling still uses the full 6,000-row file. The deduplicated table is only the E9 sensitivity set.