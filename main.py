import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np

# ── 1. LOAD ────────────────────────────────────────────────────────────────
p1 = pd.read_csv("Datasets/transitional phase 1.csv")
p2 = pd.read_csv("Datasets/transitional phase 2.csv")


print(f"Phase 1 loaded : {p1.shape}")
print(f"Phase 2 loaded : {p2.shape}")

# ── 2. ANONYMIZE NAMES ─────────────────────────────────────────────────────
p1["Name"] = ["STU_" + str(i+1).zfill(4) for i in range(len(p1))]
p2["Name"] = ["STU_" + str(i+1).zfill(4) for i in range(len(p2))]

# ── 3. PARSE DATE INTAKE ───────────────────────────────────────────────────
p1["Date Intake"] = pd.to_datetime(p1["Date Intake"], dayfirst=True)
p2["Date Intake"] = pd.to_datetime(p2["Date Intake"], dayfirst=True)

p1["Intake_Month"]  = p1["Date Intake"].dt.month
p1["Intake_Year"]   = p1["Date Intake"].dt.year
p1["Intake_Season"] = p1["Intake_Month"].map({1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4})

p2["Intake_Month"]  = p2["Date Intake"].dt.month
p2["Intake_Year"]   = p2["Date Intake"].dt.year
p2["Intake_Season"] = p2["Intake_Month"].map({1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4})

min_date = min(p1["Date Intake"].min(), p2["Date Intake"].min())
p1["Days_Since_First_Intake"] = (p1["Date Intake"] - min_date).dt.days
p2["Days_Since_First_Intake"] = (p2["Date Intake"] - min_date).dt.days

p1.drop(columns=["Date Intake"], inplace=True)
p2.drop(columns=["Date Intake"], inplace=True)

# ── 4. BUDGET MIDPOINT ─────────────────────────────────────────────────────
budget_midpoint = {"Below 10k": 8000, "10-15K": 12500, "16-20K": 18000}
p1["Budget_Midpoint"] = p1["Budget"].map(budget_midpoint)
p2["Budget_Midpoint"] = p2["Budget"].map(budget_midpoint)

# ── 5. POLITICAL PHASE NUMBER ──────────────────────────────────────────────
phase_map = {
    "Transitional phase 1st 5 months": 1,
    "Transitional phase Last 5 months": 2
}
p1["Phase_Number"] = p1["Political_Phase"].map(phase_map)
p2["Phase_Number"] = p2["Political_Phase"].map(phase_map)

# ── 6. ORDINAL MAPPING ─────────────────────────────────────────────────────
ordinal_map = {"High": 2, "Medium": 1, "Low": 0}
for col in ["Budget_Level", "Academic_Level"]:
    p1[col] = p1[col].map(ordinal_map)
    p2[col] = p2[col].map(ordinal_map)

# ── 7. BINARY MAPPING ──────────────────────────────────────────────────────
binary_map = {"Yes": 1, "No": 0}
for col in ["Researched", "Continuation"]:
    p1[col] = p1[col].map(binary_map)
    p2[col] = p2[col].map(binary_map)

# ── 8. LABEL ENCODE ────────────────────────────────────────────────────────
le_country = LabelEncoder()
le_course  = LabelEncoder()
le_phase   = LabelEncoder()
le_inst    = LabelEncoder()

for col, le in [("Country", le_country), ("Course", le_course),
                ("Political_Phase", le_phase), ("Institution", le_inst)]:
    le.fit(pd.concat([p1[col], p2[col]]).unique())
    p1[col] = le.transform(p1[col])
    p2[col] = le.transform(p2[col])

# ── 9. FEATURE ENGINEERING ─────────────────────────────────────────────────
p1["Result_Band"] = pd.cut(p1["Result"], bins=[0,2.99,3.39,4.0], labels=[0,1,2], include_lowest=True).astype(int)
p2["Result_Band"] = pd.cut(p2["Result"], bins=[0,2.99,3.39,4.0], labels=[0,1,2], include_lowest=True).astype(int)

p1["Budget_Academic_Score"] = p1["Budget_Level"] + p1["Academic_Level"]
p2["Budget_Academic_Score"] = p2["Budget_Level"] + p2["Academic_Level"]

p1["GPA_Academic_Match"] = (p1["Result_Band"] == p1["Academic_Level"]).astype(int)
p2["GPA_Academic_Match"] = (p2["Result_Band"] == p2["Academic_Level"]).astype(int)

# ── 10. SAVE CSVs ──────────────────────────────────────────────────────────
p1.to_csv("Datasets/transitional_phase_1_clean.csv", index=False)
p2.to_csv("Datasets/transitional_phase_2_clean.csv", index=False)

print("\nDone! Files saved.")
print(f"\nFinal columns ({len(p1.columns)}): {p1.columns.tolist()}")
print(p1.head(3))

# ── 11. COMPLEX VALIDATION & QUALITY REPORT ────────────────────────────────
report = []

for name, df in [("Phase 1", p1), ("Phase 2", p2)]:
    report.append(f"{'='*50}")
    report.append(f"=== {name} QUALITY REPORT ===")
    report.append(f"{'='*50}")

    report.append("\n--- Basic Info ---")
    report.append(f"Total Records     : {len(df)}")
    report.append(f"Total Features    : {len(df.columns)}")
    report.append(f"Missing Values    : {df.isnull().sum().sum()}")
    report.append(f"Duplicate Rows    : {df.duplicated().sum()}")

    report.append("\n--- Outlier Detection (IQR Method) ---")
    for col in ["Result", "Days_Since_First_Intake", "Budget_Academic_Score"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        report.append(f"{col:30s} : {len(outliers)} outliers (range {lower:.2f} - {upper:.2f})")

    report.append("\n--- GPA Summary by Budget Level ---")
    gpa_budget = df.groupby("Budget_Level")["Result"].agg(["mean", "min", "max", "count"])
    report.append(gpa_budget.to_string())

    report.append("\n--- GPA Summary by Academic Level ---")
    gpa_academic = df.groupby("Academic_Level")["Result"].agg(["mean", "min", "max", "count"])
    report.append(gpa_academic.to_string())

    report.append("\n--- Continuation Rate by Country ---")
    cont_country = df.groupby("Country")["Continuation"].agg(["sum", "count"])
    cont_country["rate_%"] = (cont_country["sum"] / cont_country["count"] * 100).round(2)
    report.append(cont_country.to_string())

    report.append("\n--- Continuation Rate by Budget Level ---")
    cont_budget = df.groupby("Budget_Level")["Continuation"].agg(["sum", "count"])
    cont_budget["rate_%"] = (cont_budget["sum"] / cont_budget["count"] * 100).round(2)
    report.append(cont_budget.to_string())
    report.append("")

# ── CROSS PHASE COMPARISON ─────────────────────────────────────────────────
report.append(f"{'='*50}")
report.append("=== CROSS PHASE DISTRIBUTION COMPARISON ===")
report.append(f"{'='*50}")

for col in ["Result", "Budget_Midpoint", "Budget_Academic_Score", "Days_Since_First_Intake"]:
    stat, p_value = stats.ks_2samp(p1[col], p2[col])
    similarity = "Similar" if p_value > 0.05 else "Different"
    report.append(f"{col:30s} : KS stat={stat:.4f}, p={p_value:.4f} → {similarity}")

report.append("\n--- Mean Comparison Phase 1 vs Phase 2 ---")
for col in ["Result", "Budget_Midpoint", "Budget_Academic_Score"]:
    report.append(f"{col:30s} : Phase1={p1[col].mean():.2f}  Phase2={p2[col].mean():.2f}")

report.append("\n--- Continuation Rate ---")
report.append(f"Phase 1 Continuation Rate : {p1['Continuation'].mean()*100:.2f}%")
report.append(f"Phase 2 Continuation Rate : {p2['Continuation'].mean()*100:.2f}%")

# ── SAVE REPORT ────────────────────────────────────────────────────────────
with open("Datasets/quality_report.txt", "w") as f:
    f.write("\n".join(report))

print("\n".join(report))
print("\n✅ Quality report saved to Datasets/quality_report.txt")