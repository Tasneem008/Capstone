import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


p1 = pd.read_csv("Datasets/transitional_phase_1_clean.csv")
p2 = pd.read_csv("Datasets/transitional_phase_2_clean.csv")

df = pd.concat([p1, p2], ignore_index=True) 

print(df.select_dtypes(include="object").columns)

# Combine both phases


print("Dataset Shape:", df.shape)


target = "Continuation"

# Remove target and all non-numeric columns
X = df.drop(columns=[target])

# Keep only numeric columns
X = X.select_dtypes(include=["number"])

y = df[target]

print("\nFeatures Used:")
print(X.columns.tolist())
print("\nNumber of features:", X.shape[1])


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
    test_size=2/3,          # 20% test, 10% validation
    random_state=42,
    stratify=y_temp
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# LOGISTIC REGRESSION


print("\nLOGISTIC REGRESSION")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# 5-Fold Cross Validation
lr_cv_scores = cross_val_score(
    lr,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\nLogistic Regression 5-Fold CV Scores:")
print(lr_cv_scores)
print("Average CV Accuracy:", lr_cv_scores.mean())

lr.fit(X_train_scaled, y_train)

val_pred_lr = lr.predict(X_val_scaled)
test_pred_lr = lr.predict(X_test_scaled)

print("\nValidation Accuracy:",
      accuracy_score(y_val, val_pred_lr))

print("Test Accuracy:",
      accuracy_score(y_test, test_pred_lr))

print("\nClassification Report")
print(classification_report(y_test, test_pred_lr))


# RANDOM FOREST



print("\nRANDOM FOREST")


rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# 5-Fold Cross Validation
rf_cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\nRandom Forest 5-Fold CV Scores:")
print(rf_cv_scores)
print("Average CV Accuracy:", rf_cv_scores.mean())

rf.fit(X_train, y_train)

val_pred_rf = rf.predict(X_val)
test_pred_rf = rf.predict(X_test)

print("\nValidation Accuracy:",
      accuracy_score(y_val, val_pred_rf))

print("Test Accuracy:",
      accuracy_score(y_test, test_pred_rf))

print("\nClassification Report")
print(classification_report(y_test, test_pred_rf))


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Validation Accuracy": [
        accuracy_score(y_val, val_pred_lr),
        accuracy_score(y_val, val_pred_rf)
    ],
    "Test Accuracy": [
        accuracy_score(y_test, test_pred_lr),
        accuracy_score(y_test, test_pred_rf)
    ],
    "5-Fold CV Accuracy": [
    lr_cv_scores.mean(),
    rf_cv_scores.mean()
   ],
    "Precision": [
        precision_score(y_test, test_pred_lr),
        precision_score(y_test, test_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, test_pred_lr),
        recall_score(y_test, test_pred_rf)
    ],
    "F1 Score": [
        f1_score(y_test, test_pred_lr),
        f1_score(y_test, test_pred_rf)
    ]
})

print("\n")

print("MODEL COMPARISON")


print(results)


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

print(importance)

importance.to_csv(
    "Datasets/random_forest_feature_importance.csv",
    index=False
)

results.to_csv(
    "Datasets/model_comparison_results.csv",
    index=False
)

lr_prob = lr.predict_proba(X_test_scaled)[:, 1] #takes only probability of yes.
rf_prob = rf.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(7,6))
plt.plot(fpr_lr, tpr_lr,
         label=f"Logistic Regression (AUC={auc_lr:.3f})")

plt.plot(fpr_rf, tpr_rf,
         label=f"Random Forest (AUC={auc_rf:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("Datasets/roc_curve_comparison.png") #saves graph.
plt.show()

ConfusionMatrixDisplay.from_estimator(
    lr,
    X_test_scaled,
    y_test,
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("Datasets/lr_confusion_matrix.png")
plt.show()

ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test,
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")
plt.savefig("Datasets/rf_confusion_matrix.png")
plt.show()

print("\nResults saved successfully.")