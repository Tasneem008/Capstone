import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

stable = pd.read_excel("Datasets/stable.xlsx")
unstable = pd.read_excel("Datasets/unstable.xlsx")
transitional = pd.read_excel("Datasets/Transitional.xlsx")
p1 = pd.read_excel("Datasets/transitional phase 1.xlsx")
p2 = pd.read_excel("Datasets/transitional phase 2.xlsx")


print("DATASET LOADING")


print(f"Stable loaded                  : {stable.shape}")
print(f"Unstable loaded                : {unstable.shape}")
print(f"Transitional loaded            : {transitional.shape}")
print(f"Transitional Phase 1 loaded    : {p1.shape}")
print(f"Transitional Phase 2 loaded    : {p2.shape}")




stable["Dataset_Type"] = "Stable"
unstable["Dataset_Type"] = "Unstable"
transitional["Dataset_Type"] = "Transitional"
p1["Dataset_Type"] = "Transitional_Phase_1"
p2["Dataset_Type"] = "Transitional_Phase_2"




df = pd.concat(
    [stable, unstable, transitional, p1, p2],
    ignore_index=True
)


print("\nMERGED DATASET")

print(f"Total records : {df.shape[0]}")
print(f"Total columns : {df.shape[1]}")



df["Name"] = [
    "STU_" + str(i + 1).zfill(4)
    for i in range(len(df))
]



df["Date Intake"] = pd.to_datetime(
    df["Date Intake"],
    dayfirst=True,
    errors="coerce"
)



df["Intake_Month"] = df["Date Intake"].dt.month

df["Intake_Year"] = df["Date Intake"].dt.year

df["Intake_Season"] = df["Intake_Month"].map({
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4
})




min_date = df["Date Intake"].min()

df["Days_Since_First_Intake"] = (
    df["Date Intake"] - min_date
).dt.days


# Date is no longer needed
df.drop(columns=["Date Intake"], inplace=True)




budget_midpoint = {
    "Below 10k": 8000,
    "10-15K": 12500,
    "16-20K": 18000
}

df["Budget_Midpoint"] = df["Budget"].map(budget_midpoint)




phase_map = {
    "Transitional": 0,
    "Transitional phase 1st 5 months": 1,
    "Transitional phase Last 5 months": 2,
    "Newly Elected": 3,
    "Unstable": 4
}

df["Phase_Number"] = df["Political_Phase"].map(phase_map)




ordinal_map = {
    "High": 2,
    "Medium": 1,
    "Low": 0
}

for col in ["Budget_Level", "Academic_Level"]:
    df[col] = df[col].map(ordinal_map)



binary_map = {
    "Yes": 1,
    "No": 0
}

for col in ["Researched", "Continuation"]:
    df[col] = df[col].map(binary_map)




le_country = LabelEncoder()
le_course = LabelEncoder()
le_phase = LabelEncoder()
le_inst = LabelEncoder()


for col, le in [
    ("Country", le_country),
    ("Course", le_course),
    ("Political_Phase", le_phase),
    ("Institution", le_inst)
]:

    df[col] = le.fit_transform(df[col].astype(str))




df["Result_Band"] = pd.cut(
    df["Result"],
    bins=[0, 2.99, 3.39, 4.0],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

df["Budget_Academic_Score"] = (
    df["Budget_Level"] +
    df["Academic_Level"]
)

df["GPA_Academic_Match"] = (
    df["Result_Band"] == df["Academic_Level"]
).astype(int)



print("\nAFTER PREPROCESSING")


print(f"Dataset shape : {df.shape}")

print("\nMissing values:")
print(df.isnull().sum())



# Because all datasets were merged before preprocessing,
# we split them back using Dataset_Type.

stable_clean = df[df["Dataset_Type"] == "Stable"].copy()
unstable_clean = df[df["Dataset_Type"] == "Unstable"].copy()
transitional_clean = df[df["Dataset_Type"] == "Transitional"].copy()
p1_clean = df[df["Dataset_Type"] == "Transitional_Phase_1"].copy()
p2_clean = df[df["Dataset_Type"] == "Transitional_Phase_2"].copy()


stable_clean.to_csv(
    "Datasets/stable_clean.csv",
    index=False
)

unstable_clean.to_csv(
    "Datasets/unstable_clean.csv",
    index=False
)

transitional_clean.to_csv(
    "Datasets/transitional_clean.csv",
    index=False
)

p1_clean.to_csv(
    "Datasets/transitional_phase_1_clean.csv",
    index=False
)

p2_clean.to_csv(
    "Datasets/transitional_phase_2_clean.csv",
    index=False
)




df.to_csv(
    "Datasets/merged_all_5_datasets_clean.csv",
    index=False
)


print("\nFILES SAVED")


print("stable_clean.csv")
print("unstable_clean.csv")
print("transitional_clean.csv")
print("transitional_phase_1_clean.csv")
print("transitional_phase_2_clean.csv")
print("merged_all_5_datasets_clean.csv")




print("\nFINAL COLUMNS")


print(f"Number of columns : {len(df.columns)}")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())




report = []

report.append("QUALITY REPORT - ALL FIVE DATASETS")




datasets = [
    ("Stable", stable_clean),
    ("Unstable", unstable_clean),
    ("Transitional", transitional_clean),
    ("Transitional Phase 1", p1_clean),
    ("Transitional Phase 2", p2_clean)
]


for name, data in datasets:

    report.append(f"{name.upper()} QUALITY REPORT")


    report.append("\n--- Basic Info ---")

    report.append(
        f"Total Records     : {len(data)}"
    )

    report.append(
        f"Total Features    : {len(data.columns)}"
    )

    report.append(
        f"Missing Values    : {data.isnull().sum().sum()}"
    )

    report.append(
        f"Duplicate Rows    : {data.duplicated().sum()}"
    )




    report.append("\n--- Outlier Detection (IQR Method) ---")

    for col in [
        "Result",
        "Days_Since_First_Intake",
        "Budget_Academic_Score"
    ]:

        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = data[
            (data[col] < lower) |
            (data[col] > upper)
        ]

        report.append(
            f"{col:30s} : "
            f"{len(outliers)} outliers "
            f"(range {lower:.2f} - {upper:.2f})"
        )


   

    report.append("\n--- GPA Summary by Budget Level ---")

    gpa_budget = data.groupby(
        "Budget_Level"
    )["Result"].agg(
        ["mean", "min", "max", "count"]
    )

    report.append(
        gpa_budget.to_string()
    )


    

    report.append("\n--- GPA Summary by Academic Level ---")

    gpa_academic = data.groupby(
        "Academic_Level"
    )["Result"].agg(
        ["mean", "min", "max", "count"]
    )

    report.append(
        gpa_academic.to_string()
    )




    report.append("\n--- Continuation Rate by Country ---")

    cont_country = data.groupby(
        "Country"
    )["Continuation"].agg(
        ["sum", "count"]
    )

    cont_country["rate_%"] = (
        cont_country["sum"] /
        cont_country["count"] *
        100
    ).round(2)

    report.append(
        cont_country.to_string()
    )



    report.append("\n--- Continuation Rate by Budget Level ---")

    cont_budget = data.groupby(
        "Budget_Level"
    )["Continuation"].agg(
        ["sum", "count"]
    )

    cont_budget["rate_%"] = (
        cont_budget["sum"] /
        cont_budget["count"] *
        100
    ).round(2)

    report.append(
        cont_budget.to_string()
    )




    report.append(
        f"\nOverall Continuation Rate : "
        f"{data['Continuation'].mean() * 100:.2f}%"
    )





report.append("\n\n")

report.append("CROSS DATASET DISTRIBUTION COMPARISON")





for name, data in datasets:

    report.append(f"\n--- {name} vs Overall Dataset ---")

    for col in [
        "Result",
        "Budget_Midpoint",
        "Budget_Academic_Score",
        "Days_Since_First_Intake"
    ]:

        stat, p_value = stats.ks_2samp(
            data[col],
            df[col]
        )

        similarity = (
            "Similar"
            if p_value > 0.05
            else "Different"
        )

        report.append(
            f"{col:30s} : "
            f"KS stat={stat:.4f}, "
            f"p={p_value:.4f} -> {similarity}"
        )




report.append("\n--- Mean Comparison ---")

for name, data in datasets:

    report.append(f"\n{name}")

    for col in [
        "Result",
        "Budget_Midpoint",
        "Budget_Academic_Score"
    ]:

        report.append(
            f"{col:30s} : "
            f"{data[col].mean():.2f}"
        )




report.append("\n--- Continuation Rate Comparison ---")

for name, data in datasets:

    report.append(
        f"{name:30s} : "
        f"{data['Continuation'].mean() * 100:.2f}%"
    )


report.append(
    f"\n{'ALL DATASETS':30s} : "
    f"{df['Continuation'].mean() * 100:.2f}%"
)



with open(
    "Datasets/quality_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report))


print("\n".join(report))

print(
    "\nQuality report saved to "
    "Datasets/quality_report.txt"
)