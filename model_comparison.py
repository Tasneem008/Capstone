import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt



stable = pd.read_csv(
    "Datasets/stable_clean.csv"
)

unstable = pd.read_csv(
    "Datasets/unstable_clean.csv"
)

transitional = pd.read_csv(
    "Datasets/transitional_clean.csv"
)

p1 = pd.read_csv(
    "Datasets/transitional_phase_1_clean.csv"
)

p2 = pd.read_csv(
    "Datasets/transitional_phase_2_clean.csv"
)


print("\nDATASETS LOADED")


print("Stable                  :", stable.shape)
print("Unstable                :", unstable.shape)
print("Transitional            :", transitional.shape)
print("Transitional Phase 1    :", p1.shape)
print("Transitional Phase 2    :", p2.shape)



df = pd.concat(
    [
        stable,
        unstable,
        transitional,
        p1,
        p2
    ],
    ignore_index=True
)


print("\nCOMBINED DATASET")


print("Dataset Shape:", df.shape)



print("\nDataset Type Distribution:")
print(df["Dataset_Type"].value_counts())



target = "Continuation"



X = df.drop(
    columns=[target]
)

y = df[target]


# Remove non-numeric columns
# This removes:
# Name
# Dataset_Type

X = X.select_dtypes(
    include=["number"]
)


print("\nFEATURES USED")


print(X.columns.tolist())

print(
    "\nNumber of features:",
    X.shape[1]
)



print("\nTARGET DISTRIBUTION")

print(y.value_counts())

print("\nTarget Percentage:")
print(
    y.value_counts(normalize=True) * 100
)



X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=2/3,
    random_state=42,
    stratify=y_temp
)


print("\nDATA SPLIT")


print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



lr = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])



rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)



print("\n")

print("LOGISTIC REGRESSION")



lr_cv_scores = cross_val_score(
    lr,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


print("\n5-Fold CV Scores:")
print(lr_cv_scores)

print(
    "Average CV Accuracy:",
    lr_cv_scores.mean()
)



lr.fit(
    X_train,
    y_train
)


val_pred_lr = lr.predict(
    X_val
)

test_pred_lr = lr.predict(
    X_test
)


print(
    "\nValidation Accuracy:",
    accuracy_score(
        y_val,
        val_pred_lr
    )
)


print(
    "Test Accuracy:",
    accuracy_score(
        y_test,
        test_pred_lr
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred_lr
    )
)



print("\n")

print("RANDOM FOREST")



rf_cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


print("\n5-Fold CV Scores:")
print(rf_cv_scores)

print(
    "Average CV Accuracy:",
    rf_cv_scores.mean()
)



rf.fit(
    X_train,
    y_train
)


val_pred_rf = rf.predict(
    X_val
)

test_pred_rf = rf.predict(
    X_test
)


print(
    "\nValidation Accuracy:",
    accuracy_score(
        y_val,
        val_pred_rf
    )
)


print(
    "Test Accuracy:",
    accuracy_score(
        y_test,
        test_pred_rf
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred_rf
    )
)



lr_prob = lr.predict_proba(
    X_test
)[:, 1]


rf_prob = rf.predict_proba(
    X_test
)[:, 1]


fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    lr_prob
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_prob
)


auc_lr = auc(
    fpr_lr,
    tpr_lr
)

auc_rf = auc(
    fpr_rf,
    tpr_rf
)



results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Validation Accuracy": [
        accuracy_score(
            y_val,
            val_pred_lr
        ),
        accuracy_score(
            y_val,
            val_pred_rf
        )
    ],

    "Test Accuracy": [
        accuracy_score(
            y_test,
            test_pred_lr
        ),
        accuracy_score(
            y_test,
            test_pred_rf
        )
    ],

    "5-Fold CV Accuracy": [
        lr_cv_scores.mean(),
        rf_cv_scores.mean()
    ],

    "Precision": [
        precision_score(
            y_test,
            test_pred_lr
        ),
        precision_score(
            y_test,
            test_pred_rf
        )
    ],

    "Recall": [
        recall_score(
            y_test,
            test_pred_lr
        ),
        recall_score(
            y_test,
            test_pred_rf
        )
    ],

    "F1 Score": [
        f1_score(
            y_test,
            test_pred_lr
        ),
        f1_score(
            y_test,
            test_pred_rf
        )
    ],

    "ROC-AUC": [
        auc_lr,
        auc_rf
    ]
})



print("\n")

print("MODEL COMPARISON")


print(
    results.to_string(
        index=False
    )
)



best_precision_model = results.loc[
    results["Precision"].idxmax(),
    "Model"
]

best_f1_model = results.loc[
    results["F1 Score"].idxmax(),
    "Model"
]

best_auc_model = results.loc[
    results["ROC-AUC"].idxmax(),
    "Model"
]


print("\n")

print("BEST MODEL ANALYSIS")


print(
    "Best Model by Precision :",
    best_precision_model
)

print(
    "Best Model by F1 Score  :",
    best_f1_model
)

print(
    "Best Model by ROC-AUC   :",
    best_auc_model
)



importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n")

print("RANDOM FOREST FEATURE IMPORTANCE")

print(
    importance.to_string(
        index=False
    )
)



importance.to_csv(
    "Datasets/random_forest_feature_importance.csv",
    index=False
)



results.to_csv(
    "Datasets/model_comparison_results.csv",
    index=False
)



plt.figure(
    figsize=(7, 6)
)


plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC={auc_lr:.3f})"
)


plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC={auc_rf:.3f})"
)


plt.plot(
    [0, 1],
    [0, 1],
    "k--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "Datasets/roc_curve_comparison.png"
)

plt.show()



ConfusionMatrixDisplay.from_estimator(
    lr,
    X_test,
    y_test,
    cmap="Blues"
)


plt.title(
    "Logistic Regression Confusion Matrix"
)


plt.savefig(
    "Datasets/lr_confusion_matrix.png"
)

plt.show()



ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test,
    cmap="Greens"
)


plt.title(
    "Random Forest Confusion Matrix"
)


plt.savefig(
    "Datasets/rf_confusion_matrix.png"
)

plt.show()



print("\n")
print("RESULTS SAVED SUCCESSFULLY")


print(
    "model_comparison_results.csv"
)

print(
    "random_forest_feature_importance.csv"
)

print(
    "roc_curve_comparison.png"
)

print(
    "lr_confusion_matrix.png"
)

print(
    "rf_confusion_matrix.png"
)